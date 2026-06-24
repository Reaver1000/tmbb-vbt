"""IFVG Sniper — vectorbt signal generator.

Ports the stateful IFVG logic from TMBB_Backtest as a pure NumPy/Python loop
that emits entry signal arrays.  vectorbt then handles all portfolio simulation.

Signal logic is identical to the backtrader version:
  Bull FVG: low[i] > high[i-2]  →  stored as dir=+1
  Bear FVG: high[i] < low[i-2]  →  stored as dir=-1
  Bull IFVG (long):  stored dir=-1, close > top + break_buffer
  Bear IFVG (short): stored dir=+1, close < bot - break_buffer
Quality filter and age expiry are applied as in TMBB_Backtest.
"""
import numpy as np
import pandas as pd

from engine import wilder_atr, TICK_SIZE, POINT_VALUE

LABEL       = 'IFVG Sniper Entry Engine (vbt)'
DESCRIPTION = 'Inverted Fair Value Gap; NumPy signal gen + vectorbt portfolio sim.'

PARAM_GRID = {
    'filter_mode': ['off', 'loose', 'balanced', 'strict'],
    'atr_len':     [7, 10, 14, 20],
    'sl_atr_mult': [0.5, 1.0, 1.5, 2.0, 2.5],
    'rr_ratio':    [1.0, 2.0, 3.0],
    'max_fvg_age': [20, 40, 60, 100],
}

DEFAULT_PARAMS = dict(
    max_hidden_fvg=120,
    max_fvg_age=60,
    min_gap_ticks=0,
    filter_mode='balanced',
    custom_min_gap_atr=0.25,
    custom_min_body_ratio=0.50,
    custom_min_range_atr=0.60,
    custom_break_atr=0.05,
    atr_len=14,
    sl_atr_mult=1.5,
    rr_ratio=3.0,
    entry_mode='confirm_close',
    max_trades_per_day=10,
    session_start_hhmm=930,
    trade_end_hhmm=1600,
    dollar_risk=100.0,
)

_FILTER_PRESETS = {
    'off':      dict(min_gap_atr=0.00, min_body_ratio=0.00, min_range_atr=0.00, break_atr=0.00),
    'loose':    dict(min_gap_atr=0.15, min_body_ratio=0.40, min_range_atr=0.40, break_atr=0.00),
    'balanced': dict(min_gap_atr=0.25, min_body_ratio=0.50, min_range_atr=0.60, break_atr=0.05),
    'strict':   dict(min_gap_atr=0.40, min_body_ratio=0.60, min_range_atr=0.85, break_atr=0.10),
}


def compute_signals(df: pd.DataFrame, params: dict) -> dict:
    """
    Compute IFVG entry/exit signal arrays for one parameter combination.

    Returns dict with keys:
        long_entries   : bool array, True on bars where a long is triggered
        short_entries  : bool array, True on bars where a short is triggered
        sl_frac        : float array, fractional SL distance from entry price
        tp_frac        : float array, fractional TP distance from entry price
        qty            : float array, contract size per bar
    """
    close  = df['Close'].values
    high   = df['High'].values
    low    = df['Low'].values
    open_  = df['Open'].values
    dt_idx = df.index

    n = len(df)
    atr_len      = int(params.get('atr_len', 14))
    max_fvg_age  = int(params.get('max_fvg_age', 60))
    max_hidden   = int(params.get('max_hidden_fvg', 120))
    min_gap_ticks= int(params.get('min_gap_ticks', 0))
    filter_mode  = params.get('filter_mode', 'balanced')
    sl_atr_mult  = float(params.get('sl_atr_mult', 1.5))
    rr_ratio     = float(params.get('rr_ratio', 3.0))
    dollar_risk  = float(params.get('dollar_risk', 100.0))
    sess_start   = int(params.get('session_start_hhmm', 930))
    trade_end    = int(params.get('trade_end_hhmm', 1600))
    max_daily    = int(params.get('max_trades_per_day', 10))

    if filter_mode in _FILTER_PRESETS:
        fp = _FILTER_PRESETS[filter_mode]
    else:
        fp = dict(
            min_gap_atr=float(params.get('custom_min_gap_atr', 0.25)),
            min_body_ratio=float(params.get('custom_min_body_ratio', 0.50)),
            min_range_atr=float(params.get('custom_min_range_atr', 0.60)),
            break_atr=float(params.get('custom_break_atr', 0.05)),
        )

    atr = wilder_atr(high, low, close, atr_len)

    long_entries  = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)
    sl_frac       = np.zeros(n, dtype=float)
    tp_frac       = np.zeros(n, dtype=float)
    qty           = np.ones(n, dtype=float)

    min_gap  = min_gap_ticks * TICK_SIZE
    raw_fvgs = []
    prev_day = None
    day_trades = 0

    for i in range(3, n):
        # --- session / day filter ---
        dt = dt_idx[i]
        day = dt.date()
        if day != prev_day:
            day_trades = 0
            prev_day = day
        hhmm = dt.hour * 100 + dt.minute
        in_session = sess_start <= hhmm < trade_end

        safe_atr = max(atr[i], TICK_SIZE)

        # --- age and expire stored FVGs ---
        for fvg in raw_fvgs:
            fvg['age'] += 1
        raw_fvgs = [f for f in raw_fvgs if f['age'] <= max_fvg_age]

        # --- detect new FVG on this bar ---
        c_range    = max(high[i] - low[i], TICK_SIZE)
        c_body     = abs(close[i] - open_[i])
        body_ratio = c_body / c_range
        range_atr  = c_range / safe_atr

        # Bull FVG: low[i] > high[i-2]
        if low[i] > high[i - 2]:
            gap = low[i] - high[i - 2]
            if gap >= min_gap:
                raw_fvgs.append(dict(
                    top=low[i], bot=high[i - 2], dir=1, age=0,
                    gap_atr=gap / safe_atr,
                    body_ratio=body_ratio,
                    range_atr=range_atr,
                ))

        # Bear FVG: high[i] < low[i-2]
        if high[i] < low[i - 2]:
            gap = low[i - 2] - high[i]
            if gap >= min_gap:
                raw_fvgs.append(dict(
                    top=low[i - 2], bot=high[i], dir=-1, age=0,
                    gap_atr=gap / safe_atr,
                    body_ratio=body_ratio,
                    range_atr=range_atr,
                ))

        # Cap FVG memory
        while len(raw_fvgs) > max_hidden:
            raw_fvgs.pop(0)

        # --- detect IFVG inversion (always consume FVG even outside session) ---
        # Matches backtrader: inversions delete the FVG on every bar,
        # but only enter a trade when in_session and under daily limit.
        break_buf = safe_atr * fp['break_atr']

        for j in reversed(range(len(raw_fvgs))):
            fvg = raw_fvgs[j]
            bull_inv = fvg['dir'] == -1 and close[i] > fvg['top'] + break_buf
            bear_inv = fvg['dir'] == 1  and close[i] < fvg['bot'] - break_buf

            if bull_inv or bear_inv:
                # Only generate a trade signal when in session and under daily limit
                if in_session and day_trades < max_daily:
                    if (fvg['gap_atr']    >= fp['min_gap_atr'] and
                            fvg['body_ratio'] >= fp['min_body_ratio'] and
                            fvg['range_atr']  >= fp['min_range_atr']):

                        sl_dist = max(atr[i] * sl_atr_mult, TICK_SIZE)
                        entry_ref = close[i]
                        sl_f = sl_dist / entry_ref
                        tp_f = sl_dist * rr_ratio / entry_ref
                        size = float(max(1, int(dollar_risk / (sl_dist * POINT_VALUE))))

                        if bull_inv:
                            long_entries[i]  = True
                        else:
                            short_entries[i] = True

                        sl_frac[i] = sl_f
                        tp_frac[i] = tp_f
                        qty[i]     = size
                        day_trades += 1

                del raw_fvgs[j]  # always consume the FVG on inversion
                break

    return dict(
        long_entries=long_entries,
        short_entries=short_entries,
        sl_frac=sl_frac,
        tp_frac=tp_frac,
        qty=qty,
    )
