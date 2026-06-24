"""Walk-forward validation for Consolidation Range strategy.

Uses focused grids derived from phase-2 winners — covers the same promising
parameter space but ~30x fewer combos, making each IS optimization fast.

Phase 2 showed:
  5min  top-10: all ADX, band_mult=1.8, sl=0.7, tp1=0.5 — very consistent
  1min  top-10: all Volatility
  15min top-10: all ADX, band_mult=2.5

Grids below concentrate on those regions whilst keeping neighbouring values
so the IS optimiser still has genuine choices to make.

Timeframes:
  5min  — 144 combos, 4 folds  (~25 min total)
  1min  — 108 combos, 4 folds  (~20 min total)
  15min — 144 combos, 4 folds  (~15 min total)
  60min — SKIPPED (only 24 trades — insufficient for WF)

Usage:
    cd <repo-root>
    python _wf_consol_range.py
"""
import sys, time
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import engine
import walkforward as wf
from strategies import consolidation_range as cr

STRAT = 'consolidation_range'

# ── Focused grids (phase-2 informed, still gives IS optimiser real choices) ───

# 5min: all phase-2 top-10 were ADX/band_mult=1.8/sl=0.7/tp1=0.5
# Expand neighbouring values to keep WF honest
GRID_5MIN = {
    'method':     ['ADX'],
    'len':        [10, 14, 20],
    'band_mult':  [1.2, 1.8, 2.5],
    'sl_mult':    [0.5, 0.7],
    'tp1_mult':   [0.3, 0.5, 0.7],
    'cooldown':   [10, 20, 30],
    'adx_thresh': [15, 17, 20],
}
# 1×3×3×2×3×3×3 = 486 combos

# 1min: all phase-2 top-10 were Volatility
GRID_1MIN = {
    'method':     ['Volatility'],
    'len':        [10, 14, 20],
    'band_mult':  [1.8, 2.5],
    'sl_mult':    [0.5, 0.7],
    'tp1_mult':   [0.5, 1.0, 2.0],
    'cooldown':   [15, 25],
    'adx_thresh': [15],
}
# 1×3×2×2×3×2×1 = 72 combos

# 15min: top results used ADX/band_mult=2.5/adx_thresh=17
GRID_15MIN = {
    'method':     ['ADX'],
    'len':        [7, 10, 14],
    'band_mult':  [1.8, 2.5],
    'sl_mult':    [0.3, 0.4, 0.5],
    'tp1_mult':   [0.5, 1.0, 2.0],
    'cooldown':   [10, 20],
    'adx_thresh': [15, 17, 20],
}
# 1×3×2×3×3×2×3 = 324 combos


TF_CONFIG = {
    '5min': {
        'file':       'MNQ_5min_stitched.csv',
        'grid':        GRID_5MIN,
        'min_trades':  20,
        'n_folds':     5,
    },
    '1min': {
        'file':       'MNQ_1min_stitched.csv',
        'grid':        GRID_1MIN,
        'min_trades':  20,
        'n_folds':     5,
    },
    '15min': {
        'file':       'MNQ_15min_stitched.csv',
        'grid':        GRID_15MIN,
        'min_trades':  10,
        'n_folds':     5,
    },
}


def run_tf(tf: str, cfg: dict) -> None:
    df = engine.load_csv(engine.DATA_DIR / cfg['file'])
    days = df.index.normalize().nunique()
    n_combos = len(wf.optimize.grid_combos(cfg['grid']))

    print()
    print('=' * 72, flush=True)
    print(f'Walk-Forward  [{tf}]  {len(df):,} bars  {days} trading days  '
          f'{n_combos} combos/fold', flush=True)
    print('=' * 72, flush=True)

    out_dir = engine.RESULTS_DIR / f'{STRAT}_{tf}_walkforward'
    t0 = time.time()

    summary = wf.run_walkforward(
        df            = df,
        strategy_stem = STRAT,
        grid          = cfg['grid'],
        base_params   = cr.DEFAULT_PARAMS,
        n_folds       = cfg['n_folds'],
        min_trades    = cfg['min_trades'],
        objective     = 'pnl_dd',
        out_dir       = out_dir,
        quiet_grid    = True,
    )

    elapsed = time.time() - t0
    print(f'\n  [{tf}] total elapsed: {elapsed:.0f}s', flush=True)


print('Consolidation Range -- Walk-Forward Validation (focused grids)', flush=True)
print('Timeframes: 5min, 1min, 15min  (60min skipped -- too few trades)', flush=True)
print(flush=True)

for tf, cfg in TF_CONFIG.items():
    run_tf(tf, cfg)

print(flush=True)
print('=' * 72, flush=True)
print('All walk-forward runs complete.', flush=True)
print('Results saved to results/consolidation_range_*_walkforward/', flush=True)
