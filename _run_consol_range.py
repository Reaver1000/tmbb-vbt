"""_run_consol_range.py — Consolidation Range (Zeiierman) multi-TF backtest.

Phase 1: default params single run on all 4 timeframes (sanity check)
Phase 2: full grid optimisation on all 4 timeframes

Data: MNQ stitched files (pre-built, 17.5 months, RTH)
Timeframes: 1min, 5min, 15min, 60min

Usage:
    cd <repo-root>
    python _run_consol_range.py
"""
import sys, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import engine
import optimize
from strategies import consolidation_range as cr

STRAT      = 'consolidation_range'
MIN_TRADES = 20

# Pre-stitched MNQ files — loaded directly (no resample needed)
TF_FILES = {
    '1min':  'MNQ_1min_stitched.csv',
    '5min':  'MNQ_5min_stitched.csv',
    '15min': 'MNQ_15min_stitched.csv',
    '60min': 'MNQ_60min_stitched.csv',
}

# Smaller grid for 1min (506k bars × full grid is 30+ min even with Numba)
# Covers the most influential params; expand after seeing initial results
GRID_1MIN = {
    'method':     ['ADX', 'Volatility'],
    'len':        [10, 14, 20],
    'band_mult':  [1.5, 1.8, 2.5],
    'sl_mult':    [0.3, 0.5, 0.7],
    'tp1_mult':   [0.5, 1.0, 2.0],
    'cooldown':   [15, 25],
    'adx_thresh': [15, 20],
}
# 2×3×3×3×3×2×2 = 648 combos

GRID_FULL = cr.PARAM_GRID
# 2×4×3×4×4×3×4 = 4608 combos  (fast on 5min/15min/60min)


def load_tf(stem: str) -> pd.DataFrame:
    return engine.load_csv(engine.DATA_DIR / stem)


def print_top(res: pd.DataFrame, tf: str, n: int = 10) -> None:
    top = res[res['valid']].head(n)
    if top.empty:
        print(f'  [{tf}] No combos with >= {MIN_TRADES} trades.')
        return
    cols = [c for c in ['method', 'len', 'band_mult', 'sl_mult', 'tp1_mult',
                         'cooldown', 'adx_thresh', 'trades', 'win%',
                         'pnl', 'pnl_dd', 'sharpe']
            if c in top.columns]
    print(f'\n  Top {len(top)} by pnl_dd  [{tf}]:')
    print(top[cols].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Default params (sanity check + first trade count)
# ─────────────────────────────────────────────────────────────────────────────
print('=' * 72)
print('Phase 1 — Default params on all timeframes')
print('  (triggers Numba JIT compile on first call — extra ~10s on 1min)')
print('=' * 72)

hdr = (f'{"TF":>6}  {"bars":>7}  {"trades":>7}  {"pnl":>10}  '
       f'{"win%":>6}  {"pnl_dd":>7}  {"sharpe":>7}  {"t/day":>6}')
print(hdr)
print('-' * len(hdr))

for tf, fname in TF_FILES.items():
    df  = load_tf(fname)
    t0  = time.time()
    m   = engine.run_single(cr, df, cr.DEFAULT_PARAMS)
    el  = time.time() - t0
    days = df.index.normalize().nunique()
    tpd  = round(m['trades'] / max(days, 1), 2)
    print(f'{tf:>6}  {len(df):>7,}  {m["trades"]:>7}  ${m["pnl"]:>9,.2f}  '
          f'{m["win%"]:>6.1f}  {m["pnl_dd"]:>7.2f}  {m["sharpe"]:>7.3f}  '
          f'{tpd:>6.2f}  ({el:.1f}s)')

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Grid optimisation
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 72)
print('Phase 2 — Grid optimisation  [objective: pnl_dd, min trades: ' +
      str(MIN_TRADES) + ']')
print('=' * 72)

RESULTS = {}

for tf, fname in TF_FILES.items():
    df   = load_tf(fname)
    grid = GRID_1MIN if tf == '1min' else GRID_FULL
    n_combos = len(optimize.grid_combos(grid))
    ckpt = engine.RESULTS_DIR / f'{STRAT}_{tf}_ckpt.csv'

    print(f'\n[{tf}]  {len(df):,} bars  |  {n_combos} combos  '
          f'|  ckpt: {ckpt.name}')
    t0  = time.time()
    res = optimize.run_grid(
        df,
        strategy_stem   = STRAT,
        grid            = grid,
        base_params     = cr.DEFAULT_PARAMS,
        min_trades      = MIN_TRADES,
        objective       = 'pnl_dd',
        checkpoint_path = ckpt,
    )
    run_dir = optimize.save_results(
        res, grid, STRAT, 'pnl_dd',
        run_dir = engine.RESULTS_DIR / f'{STRAT}_{tf}',
    )
    RESULTS[tf] = res
    elapsed = time.time() - t0
    n_valid = int(res['valid'].sum())
    print(f'  Done in {elapsed:.0f}s  |  {n_valid}/{n_combos} combos with '
          f'>= {MIN_TRADES} trades  |  saved → {run_dir.name}')
    print_top(res, tf)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 72)
print('Summary — Best combo per timeframe')
print('=' * 72)
print(f'{"TF":>6}  {"pnl_dd":>7}  {"trades":>7}  {"pnl":>10}  '
      f'{"win%":>6}  {"method":>10}  {"len":>4}  {"bmlt":>5}  '
      f'{"sl":>5}  {"tp1":>5}  {"cd":>4}  {"adxt":>5}')
print('-' * 72)
for tf, res in RESULTS.items():
    best = res[res['valid']].head(1)
    if best.empty:
        print(f'{tf:>6}  — no valid combos')
        continue
    b = best.iloc[0]
    def g(k, fmt=''):
        v = b.get(k, '?')
        return f'{v:{fmt}}' if fmt else str(v)
    print(f'{tf:>6}  {g("pnl_dd",".2f"):>7}  {g("trades",".0f"):>7}  '
          f'${b.get("pnl",0):>9,.0f}  {g("win%",".1f"):>6}  '
          f'{g("method"):>10}  {g("len"):>4}  {g("band_mult"):>5}  '
          f'{g("sl_mult"):>5}  {g("tp1_mult"):>5}  '
          f'{g("cooldown"):>4}  {g("adx_thresh"):>5}')
