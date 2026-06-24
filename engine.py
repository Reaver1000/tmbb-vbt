"""TMBB_VBT — vectorbt-based backtest engine.

Drop-in replacement for TMBB_Backtest's engine.py but orders of magnitude faster:
  - Signal generation: pure NumPy/pandas loops (no backtrader event overhead)
  - Portfolio simulation: vectorbt runs all parameter combos simultaneously

Data files are shared with TMBB_Backtest (same data/ folder path).
"""
import os
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

ROOT = Path(__file__).parent

# Data directory: set TMBB_DATA_DIR env var to override.
# Default assumes a sibling TMBB_Backtest/data/ folder (original project layout).
# Alternatively, place a data/ folder next to this file and it will be used automatically.
_local_data   = ROOT / 'data'
_default_data = ROOT.parent / 'TMBB_Backtest' / 'data'
DATA_DIR = (Path(os.environ['TMBB_DATA_DIR']) if 'TMBB_DATA_DIR' in os.environ
            else _local_data if _local_data.exists()
            else _default_data)
RESULTS_DIR = ROOT / 'results'
STRATS_DIR  = ROOT / 'strategies'

POINT_VALUE = 2.0    # MNQ: $2 per point
COMMISSION  = 0.62   # $ per contract per side
MARGIN      = 50.0
DEFAULT_CASH = 10_000.0
TICK_SIZE   = 0.25


# ── Data ──────────────────────────────────────────────────────────────────────

def load_csv(path) -> pd.DataFrame:
    """Load OHLCV CSV — handles both headered and NT8 raw formats."""
    path = Path(path)
    with open(path) as f:
        first = f.readline()
    sep = ';' if first.count(';') > first.count(',') else ','
    has_header = any(ch.isalpha() for ch in first.split(sep)[0])
    if has_header:
        df = pd.read_csv(path, sep=sep)
        df.columns = [c.strip().capitalize() for c in df.columns]
        dt_col = df.columns[0]
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.rename(columns={dt_col: 'dt'})
    else:
        df = pd.read_csv(path, sep=sep, header=None,
                         names=['dt', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['dt'] = pd.to_datetime(df['dt'].str.strip(), format='%Y%m%d %H%M%S')
    df = df.set_index('dt').sort_index()
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 1-min OHLCV to any higher timeframe."""
    r = df.resample(freq).agg(
        Open=('Open', 'first'), High=('High', 'max'),
        Low=('Low', 'min'),   Close=('Close', 'last'),
        Volume=('Volume', 'sum'),
    ).dropna(subset=['Open'])
    return r


def list_data_files():
    DATA_DIR.mkdir(exist_ok=True)
    return sorted(DATA_DIR.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)


# ── Indicators ────────────────────────────────────────────────────────────────

def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int) -> np.ndarray:
    """Wilder's ATR matching Pine Script ta.atr() and backtrader ATR."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
              np.maximum(np.abs(high[1:] - close[:-1]),
                         np.abs(low[1:] - close[:-1])))
    atr = np.zeros(n)
    if n <= period:
        return atr
    atr[period - 1] = tr[:period].mean()
    k = (period - 1) / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * k + tr[i] * (1 - k)
    return atr


# ── Strategy registry ─────────────────────────────────────────────────────────

def get_strategy(stem: str):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(f'strategies.{stem}')


# ── Metrics ───────────────────────────────────────────────────────────────────

def portfolio_metrics(pf: 'vbt.Portfolio', label: str = '',
                      cash: float = None) -> dict:
    """Extract standardised metrics from a vectorbt Portfolio."""
    trades = pf.trades.records_readable
    n = len(trades)
    if n == 0:
        return {'trades': 0, 'pnl': 0.0, 'win%': 0.0, 'pnl_dd': 0.0,
                'sharpe': 0.0, 'profit_factor': 0.0, 'maxdd_$': 0.0}

    pnls  = trades['PnL'].values
    wins  = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    # VBT P&L is in price_pts × contracts; multiply by POINT_VALUE to get dollars
    total_pnl   = float(pnls.sum()) * POINT_VALUE
    gross_w     = float(wins.sum()) * POINT_VALUE
    gross_l     = float(-losses.sum()) * POINT_VALUE

    # max_drawdown() returns a negative fraction in vbt 1.x (e.g. -0.4 = 40% DD)
    # Use the passed cash value (not pf.init_cash) so LARGE_CASH doesn't distort maxdd_$
    _cash = cash if cash is not None else float(pf.init_cash)
    maxdd_abs = abs(float(pf.max_drawdown()) * _cash) * POINT_VALUE

    return {
        'trades':        n,
        'pnl':           round(total_pnl, 2),
        'win%':          round(len(wins) / n * 100, 1),
        'profit_factor': round(gross_w / gross_l, 2) if gross_l > 0 else (np.inf if gross_w > 0 else 0.0),
        'expectancy':    round(float(pnls.mean()) * POINT_VALUE, 2),
        'maxdd_$':       round(maxdd_abs, 2),
        'pnl_dd':        round(total_pnl / max(maxdd_abs, 1.0), 2),
        'sharpe':        round(float(pf.sharpe_ratio()), 3),
    }


# ── Single run ────────────────────────────────────────────────────────────────

LARGE_CASH = 10_000_000  # prevents VBT from size-reducing on margin instruments

def run_single(mod, df: pd.DataFrame, params: dict,
               cash: float = LARGE_CASH) -> dict:
    """Run one backtest. Returns metrics dict."""
    signals = mod.compute_signals(df, params)
    close   = df['Close']

    # Commission as fraction of price (approximate: $0.62 / avg_price / point_value)
    fee_frac = COMMISSION / (close.mean() * POINT_VALUE)
    idx = close.index

    # Pass high/low so VBT checks stops intrabar (matches backtrader bracket behaviour).
    # stop_exit_price='stop' fills at the stop level, not the bar close.
    pf = vbt.Portfolio.from_signals(
        close=close,
        high=df['High'],
        low=df['Low'],
        entries=pd.Series(signals['long_entries'],  index=idx),
        short_entries=pd.Series(signals['short_entries'], index=idx),
        exits=pd.Series(False, index=idx),
        short_exits=pd.Series(False, index=idx),
        sl_stop=pd.Series(signals['sl_frac'],  index=idx),
        tp_stop=pd.Series(signals['tp_frac'],  index=idx),
        stop_exit_price='StopLimit',     # fills at stop level exactly (StopMarket fills at bar close)
        upon_opposite_entry='ignore',
        size=signals.get('qty', 1),
        size_type='amount',
        fees=fee_frac,
        init_cash=cash,
        freq=pd.infer_freq(idx) or '5min',
    )
    return portfolio_metrics(pf, cash=cash)
