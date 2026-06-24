"""Walk-forward validation for TMBB_VBT.

Anchored-expanding IS approach:
  - Data is divided into n_folds equal OOS windows
  - Fold k IS: all data before window k
  - Fold k OOS: window k
  - Re-optimises the full PARAM_GRID on each IS window
  - Tests winning params on the following OOS window

This is the honest test: each OOS period is genuinely unseen during optimisation.
Fold 1 has the smallest IS (min_oos_days of data), fold n has the most.
Minimum IS: min_oos_days (at least one OOS window worth of data).
"""
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import engine
import optimize


def make_folds(df: pd.DataFrame, n_folds: int):
    """
    Anchored-expanding IS folds.

    Divides the full date range into n_folds equal OOS windows.
    Fold k: IS = everything before window k, OOS = window k.
    Fold 0 is skipped (no IS data before the first window).

    Returns list of (is_df, oos_df, fold_label).
    """
    days = sorted(df.index.normalize().unique())
    n = len(days)
    oos_size = max(1, n // n_folds)

    folds = []
    for k in range(1, n_folds):           # k=0 skipped — no IS before first window
        oos_start = k * oos_size
        oos_end   = min((k + 1) * oos_size, n)
        if oos_start >= n:
            break
        is_set  = set(days[:oos_start])
        oos_set = set(days[oos_start:oos_end])
        is_df   = df[df.index.normalize().isin(is_set)]
        oos_df  = df[df.index.normalize().isin(oos_set)]
        label   = (f'{days[0]}→{days[oos_start - 1]} '
                   f'OOS:{days[oos_start]}→{days[min(oos_end, n) - 1]}')
        folds.append((is_df, oos_df, label))
    return folds


def run_walkforward(df: pd.DataFrame,
                    strategy_stem: str,
                    grid: dict,
                    base_params: dict,
                    n_folds: int      = 5,
                    min_trades: int   = 10,
                    objective: str    = 'pnl_dd',
                    out_dir: Path     = None,
                    quiet_grid: bool  = True) -> pd.DataFrame:
    """
    Run anchored walk-forward validation.

    For each fold: optimise on IS → test best params on OOS.
    Returns summary DataFrame with one row per fold.
    Saves wf_summary.csv to out_dir.
    """
    folds = make_folds(df, n_folds)
    if not folds:
        raise ValueError(f"No folds possible with {n_folds} folds on {len(df)} bars.")

    n_combos = len(optimize.grid_combos(grid))
    print(f"\n  Walk-forward: {len(folds)} folds  |  {n_combos} combos/IS  "
          f"|  objective={objective}  |  min_trades={min_trades}", flush=True)

    rows = []
    for fold_num, (is_df, oos_df, label) in enumerate(folds, 1):
        is_days  = is_df.index.normalize().nunique()
        oos_days = oos_df.index.normalize().nunique()
        print(f"\n  ─── Fold {fold_num}/{len(folds)} ───", flush=True)
        print(f"  IS:  {is_df.index[0].date()} → {is_df.index[-1].date()}  "
              f"({is_days} days, {len(is_df):,} bars)", flush=True)
        print(f"  OOS: {oos_df.index[0].date()} → {oos_df.index[-1].date()}  "
              f"({oos_days} days, {len(oos_df):,} bars)", flush=True)

        # ── IS optimisation ───────────────────────────────────────────────────
        t0 = time.time()
        res = optimize.run_grid(
            is_df, strategy_stem, grid, base_params,
            min_trades=min_trades, objective=objective, quiet=quiet_grid,
        )
        is_elapsed = time.time() - t0
        best_valid = res[res['valid']]

        if best_valid.empty:
            print(f"  SKIP: no IS combo with >= {min_trades} trades  ({is_elapsed:.0f}s)", flush=True)
            rows.append({'fold': fold_num, 'is_elapsed': round(is_elapsed),
                         'note': 'no_valid_IS'})
            continue

        bp = {k: best_valid.iloc[0][k] for k in grid}
        bp = {k: (v.item() if hasattr(v, 'item') else v) for k, v in bp.items()}
        is_m = best_valid.iloc[0]
        print(f"  IS  best: {bp}", flush=True)
        print(f"  IS  metrics: pnl=${is_m['pnl']:,.0f}  "
              f"trades={int(is_m['trades'])}  win%={is_m['win%']:.1f}  "
              f"pnl_dd={is_m['pnl_dd']:.2f}  ({is_elapsed:.0f}s)", flush=True)

        # ── OOS single run ────────────────────────────────────────────────────
        mod        = engine.get_strategy(strategy_stem)
        oos_params = {**base_params, **bp}
        oos_m      = engine.run_single(mod, oos_df, oos_params)
        verdict    = 'PASS' if oos_m['pnl'] > 0 else 'FAIL'
        print(f"  OOS metrics: pnl=${oos_m['pnl']:,.0f}  "
              f"trades={oos_m['trades']}  win%={oos_m['win%']:.1f}  "
              f"pnl_dd={oos_m['pnl_dd']:.2f}  sharpe={oos_m['sharpe']:.3f}  "
              f"[{verdict}]", flush=True)

        rows.append({
            'fold':          fold_num,
            'is_start':      str(is_df.index[0].date()),
            'is_end':        str(is_df.index[-1].date()),
            'oos_start':     str(oos_df.index[0].date()),
            'oos_end':       str(oos_df.index[-1].date()),
            'is_days':       is_days,
            'oos_days':      oos_days,
            **{f'p_{k}': v for k, v in bp.items()},
            'is_pnl':        round(is_m['pnl'], 2),
            'is_trades':     int(is_m['trades']),
            'is_pnl_dd':     round(is_m['pnl_dd'], 2),
            'oos_pnl':       round(oos_m['pnl'], 2),
            'oos_trades':    oos_m['trades'],
            'oos_win%':      oos_m['win%'],
            'oos_pnl_dd':    oos_m['pnl_dd'],
            'oos_sharpe':    oos_m['sharpe'],
            'oos_maxdd_$':   oos_m['maxdd_$'],
            'is_elapsed':    round(is_elapsed),
            'pass':          verdict == 'PASS',
        })

    if not rows:
        print("\n  No completed folds.")
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    completed = summary[summary.get('note', pd.Series([''] * len(summary))) != 'no_valid_IS'] \
        if 'note' in summary.columns else summary
    # re-filter to rows that have oos_pnl
    completed = summary[summary['fold'].isin(
        [r['fold'] for r in rows if 'oos_pnl' in r]
    )]

    total_oos  = completed['oos_pnl'].sum() if not completed.empty else 0.0
    n_pass     = int(completed['pass'].sum()) if not completed.empty else 0
    n_complete = len(completed)
    consistent = n_pass == n_complete and n_complete > 0

    print("\n  ═══ Walk-Forward Summary " + "═" * 45)
    if not completed.empty:
        show_cols = ['fold', 'oos_start', 'oos_end', 'oos_days',
                     'oos_pnl', 'oos_trades', 'oos_win%', 'oos_pnl_dd', 'pass']
        show_cols = [c for c in show_cols if c in completed.columns]
        print(completed[show_cols].to_string(index=False))
    print(f"\n  Total OOS P&L : ${total_oos:,.2f}")
    print(f"  Folds passed  : {n_pass}/{n_complete}")
    print(f"  Consistent    : {'YES ✓' if consistent else 'NO — likely overfit'}")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / 'wf_summary.csv', index=False)
        print(f"  Saved → {out_dir / 'wf_summary.csv'}")

    return summary
