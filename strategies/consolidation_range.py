"""Consolidation Range with Signals (Zeiierman) — vectorbt signal generator.

Faithful line-by-line port of the PineScript v6 indicator.

Detection methods:
  ADX        — ranging when ADX < threshold
  Volatility — ranging when std/var/ATR all below their rolling averages

Signal logic (Pine → Python mapping):
  rngfilt steps up while rangeVisible   →  long entry
  rngfilt steps down while rangeVisible →  short entry
  SL/TP anchored to rngfilt ± r*mult, expressed as fraction of entry close

Range filter simplification (verified against Pine source):
  Pine's pricejump(prev) + rangefilter(hhjump, lljump, prev) reduce to:
    if |close - prev| >= r:  rngfilt = close   (jump to price)
    else:                     rngfilt = prev    (no movement)
  The pricejump calls only matter in the hhAbove/llBelow branches where they
  evaluate to close anyway (see inline proof in comments).
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfilter_zi

try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    _NUMBA = False

from engine import wilder_atr

LABEL       = 'Consolidation Range with Signals (Zeiierman)'
DESCRIPTION = 'ADX/Volatility range detection + step-filter breakout signals.'

DEFAULT_PARAMS = dict(
    method       = 'ADX',
    len          = 10,
    band_mult    = 1.8,
    cooldown     = 20,
    sl_mult      = 0.4,
    tp1_mult     = 0.5,
    adx_thresh   = 17,
    adx_smooth   = 10,
    vol_mult_std = 0.8,
    vol_mult_var = 0.8,
    vol_mult_atr = 0.9,
)

PARAM_GRID = {
    'method':     ['ADX', 'Volatility'],
    'len':        [7, 10, 14, 20],
    'band_mult':  [1.2, 1.8, 2.5],
    'sl_mult':    [0.3, 0.4, 0.5, 0.7],
    'tp1_mult':   [0.5, 1.0, 1.5, 2.0],
    'cooldown':   [10, 20, 30],
    'adx_thresh': [15, 17, 20, 25],
}


# ── Wilder's RMA via scipy (100x faster than Python loop) ────────────────────

def _rma(src: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.rma() — EMA with alpha=1/length; init = SMA of first `length` values."""
    n = len(src)
    out = np.zeros(n)
    if n < length:
        return out
    init_val = float(np.mean(src[:length]))
    out[length - 1] = init_val
    if n > length:
        alpha = 1.0 / length
        b = np.array([alpha])
        a = np.array([1.0, -(1.0 - alpha)])
        zi = lfilter_zi(b, a) * init_val
        out[length:], _ = lfilter(b, a, src[length:], zi=zi)
    return out


# ── DMI / ADX (Pine ta.dmi) ──────────────────────────────────────────────────

def _dmi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         di_len: int, adx_smooth: int):
    """Returns (plus, minus, adx) arrays matching Pine ta.dmi(di_len, adx_smooth)."""
    n = len(high)

    # True Range (vectorized)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
              np.maximum(np.abs(high[1:] - close[:-1]),
                         np.abs(low[1:]  - close[:-1])))

    # Directional Movement (vectorized)
    up = np.zeros(n)
    dn = np.zeros(n)
    up[1:] = high[1:] - high[:-1]
    dn[1:] = low[:-1] - low[1:]
    dm_p = np.where((up > dn) & (up > 0.0), up, 0.0)
    dm_m = np.where((dn > up) & (dn > 0.0), dn, 0.0)

    tr_rma  = _rma(tr,   di_len)
    dmp_rma = _rma(dm_p, di_len)
    dmm_rma = _rma(dm_m, di_len)

    with np.errstate(divide='ignore', invalid='ignore'):
        plus  = np.where(tr_rma > 0.0, 100.0 * dmp_rma / tr_rma, 0.0)
        minus = np.where(tr_rma > 0.0, 100.0 * dmm_rma / tr_rma, 0.0)
        dsum  = plus + minus
        dx    = np.where(dsum > 0.0, 100.0 * np.abs(plus - minus) / dsum, 0.0)
    adx   = _rma(dx, adx_smooth)
    return plus, minus, adx


# ── Inner signal loop (Numba-JIT or pure Python fallback) ────────────────────

if _NUMBA:
    from numba import njit

    @njit(cache=True)
    def _signal_loop(close, r_arr, method_detected,
                     cooldown, sl_mult, tp1_mult):
        n = len(close)
        long_entries  = np.zeros(n)
        short_entries = np.zeros(n)
        sl_frac       = np.zeros(n)
        tp_frac       = np.zeros(n)

        rngfilt_prev  = close[0]
        range_visible = False
        prev_hi       = -1e30   # sentinel (na → very negative)
        prev_lo       =  1e30   # sentinel (na → very positive)
        last_sig_bar  = -1      # -1 = na

        for i in range(1, n):
            r  = r_arr[i]
            c  = close[i]
            md = method_detected[i]

            # ── Range filter ─────────────────────────────────────────────────
            # Pine pricejump + rangefilter simplify to:
            #   hhAbove (c >= prev+r) → hhjump = prev+|c-prev| = c  (c>prev here)
            #   llBelow (c <= prev-r) → lljump = prev-|c-prev| = c  (c<prev here)
            #   else → step1 always returns prev (hhTooClose/llTooClose always true)
            if c >= rngfilt_prev + r or c <= rngfilt_prev - r:
                rngfilt = c
            else:
                rngfilt = rngfilt_prev

            step_up   = rngfilt > rngfilt_prev
            step_down = rngfilt < rngfilt_prev
            hiband = rngfilt + r
            loband = rngfilt - r

            # ── rangeVisible state machine ────────────────────────────────────
            if md:
                range_visible = True
                prev_hi = hiband
                prev_lo = loband
            elif prev_hi > -1e29:   # prev_hi is set (not na)
                # Pine: if not methodDetected and (close[1] > prev_hi or close[1] < prev_lo)
                # close[1] in Pine = close of previous bar = close[i-1] in Python
                if close[i - 1] > prev_hi or close[i - 1] < prev_lo:
                    range_visible = False

            # ── Cooldown ──────────────────────────────────────────────────────
            can_trigger = last_sig_bar < 0 or (i - last_sig_bar) >= cooldown

            # ── Long ──────────────────────────────────────────────────────────
            if range_visible and step_up and can_trigger:
                sl_price = rngfilt - r * sl_mult
                tp_price = rngfilt + r * tp1_mult
                sl_f = (c - sl_price) / c if c > 0.0 else 0.0
                tp_f = (tp_price - c)  / c if c > 0.0 else 0.0
                if sl_f > 0.0 and tp_f > 0.0:
                    long_entries[i] = 1.0
                    sl_frac[i]      = sl_f
                    tp_frac[i]      = tp_f
                    last_sig_bar    = i

            # ── Short ─────────────────────────────────────────────────────────
            elif range_visible and step_down and can_trigger:
                sl_price = rngfilt + r * sl_mult
                tp_price = rngfilt - r * tp1_mult
                sl_f = (sl_price - c) / c if c > 0.0 else 0.0
                tp_f = (c - tp_price) / c if c > 0.0 else 0.0
                if sl_f > 0.0 and tp_f > 0.0:
                    short_entries[i] = 1.0
                    sl_frac[i]       = sl_f
                    tp_frac[i]       = tp_f
                    last_sig_bar     = i

            rngfilt_prev = rngfilt

        return long_entries, short_entries, sl_frac, tp_frac

else:
    def _signal_loop(close, r_arr, method_detected,
                     cooldown, sl_mult, tp1_mult):
        n = len(close)
        long_entries  = np.zeros(n)
        short_entries = np.zeros(n)
        sl_frac       = np.zeros(n)
        tp_frac       = np.zeros(n)

        rngfilt_prev  = close[0]
        range_visible = False
        prev_hi       = np.nan
        prev_lo       = np.nan
        last_sig_bar  = -1

        for i in range(1, n):
            r  = r_arr[i]
            c  = close[i]
            md = bool(method_detected[i])

            if c >= rngfilt_prev + r or c <= rngfilt_prev - r:
                rngfilt = c
            else:
                rngfilt = rngfilt_prev

            step_up   = rngfilt > rngfilt_prev
            step_down = rngfilt < rngfilt_prev
            hiband = rngfilt + r
            loband = rngfilt - r

            if md:
                range_visible = True
                prev_hi = hiband
                prev_lo = loband
            elif not np.isnan(prev_hi):
                if close[i - 1] > prev_hi or close[i - 1] < prev_lo:
                    range_visible = False

            can_trigger = last_sig_bar < 0 or (i - last_sig_bar) >= cooldown

            if range_visible and step_up and can_trigger:
                sl_price = rngfilt - r * sl_mult
                tp_price = rngfilt + r * tp1_mult
                sl_f = (c - sl_price) / c if c > 0.0 else 0.0
                tp_f = (tp_price - c)  / c if c > 0.0 else 0.0
                if sl_f > 0.0 and tp_f > 0.0:
                    long_entries[i] = 1.0
                    sl_frac[i]      = sl_f
                    tp_frac[i]      = tp_f
                    last_sig_bar    = i
            elif range_visible and step_down and can_trigger:
                sl_price = rngfilt + r * sl_mult
                tp_price = rngfilt - r * tp1_mult
                sl_f = (sl_price - c) / c if c > 0.0 else 0.0
                tp_f = (c - tp_price) / c if c > 0.0 else 0.0
                if sl_f > 0.0 and tp_f > 0.0:
                    short_entries[i] = 1.0
                    sl_frac[i]       = sl_f
                    tp_frac[i]       = tp_f
                    last_sig_bar     = i

            rngfilt_prev = rngfilt

        return long_entries, short_entries, sl_frac, tp_frac


# ── Main entry point ─────────────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame, params: dict) -> dict:
    """
    Compute Consolidation Range entry/exit signals.

    Returns dict: long_entries, short_entries, sl_frac, tp_frac, qty
    sl_frac / tp_frac are positive fractions of entry close price.
    """
    close = df['Close'].values
    high  = df['High'].values
    low   = df['Low'].values
    n     = len(df)

    method       = params.get('method', 'ADX')
    per          = int(params.get('len', 10))
    band_mult    = float(params.get('band_mult', 1.8))
    cooldown     = int(params.get('cooldown', 20))
    sl_mult      = float(params.get('sl_mult', 0.4))
    tp1_mult     = float(params.get('tp1_mult', 0.5))
    adx_thresh   = float(params.get('adx_thresh', 17))
    adx_smooth   = int(params.get('adx_smooth', 10))
    vol_mult_std = float(params.get('vol_mult_std', 0.8))
    vol_mult_var = float(params.get('vol_mult_var', 0.8))
    vol_mult_atr = float(params.get('vol_mult_atr', 0.9))

    # ── ADX detection ────────────────────────────────────────────────────────
    _, _, adx = _dmi(high, low, close, per, adx_smooth)
    is_adx = (adx < adx_thresh).astype(np.float64)

    # ── Volatility compression detection ──────────────────────────────────────
    # Pine: logret = math.log(close / close[1])
    logret = np.zeros(n)
    logret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-10))
    s = pd.Series(logret)

    std_now = s.rolling(per, min_periods=per).std(ddof=0).fillna(0.0).values
    std_avg = pd.Series(std_now).rolling(per, min_periods=per).mean().fillna(0.0).values
    var_now = s.rolling(per, min_periods=per).var(ddof=0).fillna(0.0).values
    var_avg = pd.Series(var_now).rolling(per, min_periods=per).mean().fillna(0.0).values
    atr_now = wilder_atr(high, low, close, per)
    atr_avg = pd.Series(atr_now).rolling(per, min_periods=per).mean().fillna(0.0).values

    # Compression: all three metrics below their own SMA × multiplier
    # When avg=0 the condition is 0 < 0 → False (safely inactive during warmup)
    is_vol = (
        (std_avg > 0) & (std_now < std_avg * vol_mult_std) &
        (var_avg > 0) & (var_now < var_avg * vol_mult_var) &
        (atr_avg > 0) & (atr_now < atr_avg * vol_mult_atr)
    ).astype(np.float64)

    method_detected = is_adx if method == 'ADX' else is_vol

    # ── r (range step size) ──────────────────────────────────────────────────
    # Pine: diff = math.abs(high - low[1])   →  abs(high[i] - low[i-1])
    #       r    = ta.sma(2.618 * diff, 2000) * band_mult
    diff = np.empty(n)
    diff[0]  = abs(high[0] - low[0])           # no prev bar, use current range
    diff[1:] = np.abs(high[1:] - low[:-1])
    r_arr = (pd.Series(2.618 * diff)
             .rolling(2000, min_periods=1)
             .mean()
             .values) * band_mult

    # ── Stateful signal loop (JIT-compiled if numba available) ───────────────
    long_e, short_e, sl_f, tp_f = _signal_loop(
        close, r_arr, method_detected,
        float(cooldown), sl_mult, tp1_mult,
    )

    return dict(
        long_entries  = long_e.astype(bool),
        short_entries = short_e.astype(bool),
        sl_frac       = sl_f,
        tp_frac       = tp_f,
        qty           = np.ones(n, dtype=float),
    )
