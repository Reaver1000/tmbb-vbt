"""TMBB_VBT — vectorbt grid optimizer.

Pipeline approach:
  For each chunk of combos:
    1. Generate signals (NumPy)
    2. Run VBT simulation on that chunk
    3. Append rows to checkpoint CSV
    4. Free memory and move to next chunk

  Memory stays bounded regardless of total combo count.
  Checkpointing allows resume after crash.
"""
import itertools
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import vectorbt as vbt

import engine

RESULTS_DIR = engine.RESULTS_DIR


def grid_combos(grid: dict) -> list:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def _combo_key(combo: dict, grid_keys: list) -> tuple:
    return tuple(combo[k] for k in grid_keys)


def run_grid(df: pd.DataFrame, strategy_stem: str, grid: dict,
             base_params: dict, cash: float = engine.LARGE_CASH,
             min_trades: int = 10, objective: str = 'pnl_dd',
             quiet: bool = False,
             checkpoint_path: Path = None) -> pd.DataFrame:
    """
    Run every PARAM_GRID combination and return a ranked results DataFrame.

    Processes combos in pipeline chunks:
      generate signals → run VBT → save checkpoint → free memory → repeat

    Pass checkpoint_path to enable resume-on-crash. If the file already
    exists, already-completed combos are skipped automatically.
    """
    mod    = engine.get_strategy(strategy_stem)
    combos = grid_combos(grid)
    if not combos:
        raise ValueError('Empty parameter grid')

    n_combos  = len(combos)
    grid_keys = list(grid.keys())
    close     = df['Close']
    n_bars    = len(close)
    pv        = engine.POINT_VALUE
    fee_frac  = engine.COMMISSION / (close.mean() * pv)
    idx       = close.index
    freq_str  = pd.infer_freq(idx) or '5min'

    # Auto-size chunk so signal arrays stay under ~1 GB
    # 4 arrays × n_bars × 8 bytes per combo
    chunk_size = max(50, int(1e9 / (n_bars * 8 * 4)))

    # --- resume support ---
    done_keys = set()
    rows = []
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = pd.read_csv(checkpoint_path)
        rows = ckpt.to_dict('records')
        for r in rows:
            done_keys.add(_combo_key(r, grid_keys))
        if not quiet:
            print(f'  Resuming: {len(done_keys)} combos already done, '
                  f'{n_combos - len(done_keys)} remaining')

    pending = [c for c in combos if _combo_key(c, grid_keys) not in done_keys]
    n_pending = len(pending)

    t0 = time.time()
    n_chunks = (n_pending + chunk_size - 1) // chunk_size
    if not quiet:
        print(f'  {n_pending}/{n_combos} combos  |  chunk={chunk_size}  |  '
              f'{n_chunks} chunk(s)  |  bars={n_bars:,}')

    for chunk_start in range(0, n_pending, chunk_size):
        chunk_end   = min(chunk_start + chunk_size, n_pending)
        chunk       = pending[chunk_start:chunk_end]
        chunk_n     = len(chunk)

        # --- generate signals for this chunk ---
        long_list, short_list, sl_list, tp_list = [], [], [], []
        valid_chunk = []
        sig_errors  = []

        for combo in chunk:
            params = {**base_params, **combo}
            try:
                sig = mod.compute_signals(df, params)
                long_list.append(sig['long_entries'])
                short_list.append(sig['short_entries'])
                sl_list.append(sig['sl_frac'])
                tp_list.append(sig['tp_frac'])
                valid_chunk.append(combo)
            except Exception as e:
                sig_errors.append((combo, str(e)[:80]))
                rows.append({**combo, 'trades': 0, 'error': str(e)[:60]})

        if not quiet:
            done_so_far = chunk_end
            pct = done_so_far / n_pending * 100
            elapsed = time.time() - t0
            print(f'\r  signals {done_so_far}/{n_pending} ({pct:.0f}%)  '
                  f'[{elapsed:.0f}s]  chunk {chunk_start//chunk_size + 1}/{n_chunks}',
                  end='', flush=True)

        if not valid_chunk:
            continue

        # --- run VBT on this chunk ---
        long_2d  = pd.DataFrame(np.column_stack(long_list),  index=idx).astype(bool)
        short_2d = pd.DataFrame(np.column_stack(short_list), index=idx).astype(bool)
        sl_2d    = pd.DataFrame(np.column_stack(sl_list),    index=idx)
        tp_2d    = pd.DataFrame(np.column_stack(tp_list),    index=idx)

        try:
            pf = vbt.Portfolio.from_signals(
                close=close,
                high=df['High'],
                low=df['Low'],
                entries=long_2d,
                short_entries=short_2d,
                exits=pd.DataFrame(False, index=idx, columns=long_2d.columns),
                short_exits=pd.DataFrame(False, index=idx, columns=long_2d.columns),
                sl_stop=sl_2d,
                tp_stop=tp_2d,
                stop_exit_price='StopLimit',
                upon_opposite_entry='ignore',
                size=1,
                size_type='amount',
                fees=fee_frac,
                init_cash=cash,
                freq=freq_str,
            )
        except Exception as e:
            # VBT chunk failed — record all as errors and continue
            for combo in valid_chunk:
                rows.append({**combo, 'trades': 0, 'error': f'vbt:{str(e)[:50]}'})
            if not quiet:
                print(f'\n  WARN chunk VBT error: {e}')
            continue

        # --- extract metrics ---
        for col_idx, combo in enumerate(valid_chunk):
            col = pf.iloc[col_idx] if hasattr(pf, 'iloc') else pf
            try:
                trades_rec = col.trades.records_readable
                n_trades   = len(trades_rec)
                pnls = trades_rec['PnL'].values if n_trades > 0 else np.array([])
                wins = pnls[pnls > 0]
                loss = pnls[pnls <= 0]
                total_pnl = float(pnls.sum()) * pv if n_trades else 0.0
                maxdd_abs = abs(float(col.max_drawdown() * cash)) * pv
                gross_w   = float(wins.sum()) * pv if len(wins) else 0.0
                gross_l   = float(-loss.sum()) * pv if len(loss) else 0.0
                row = {
                    **combo,
                    'trades':        n_trades,
                    'pnl':           round(total_pnl, 2),
                    'win%':          round(len(wins) / n_trades * 100, 1) if n_trades else 0.0,
                    'profit_factor': round(gross_w / gross_l, 2) if gross_l > 0 else 0.0,
                    'expectancy':    round(float(pnls.mean()) * pv, 2) if n_trades else 0.0,
                    'maxdd_$':       round(maxdd_abs, 2),
                    'pnl_dd':        round(total_pnl / max(maxdd_abs, 1.0), 2),
                    'sharpe':        round(float(col.sharpe_ratio()), 3),
                }
            except Exception as e:
                row = {**combo, 'trades': 0, 'error': str(e)[:60]}
            rows.append(row)

        # --- checkpoint after every chunk ---
        if checkpoint_path is not None:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

        # free chunk memory explicitly
        del long_list, short_list, sl_list, tp_list
        del long_2d, short_2d, sl_2d, tp_2d, pf

    if not quiet:
        print()

    res = pd.DataFrame(rows)
    res['valid'] = res['trades'] >= min_trades
    sort_key = objective if objective in res.columns else 'pnl_dd'
    res = res.sort_values(['valid', sort_key], ascending=[False, False]).reset_index(drop=True)

    elapsed = time.time() - t0
    if not quiet:
        n_valid_count = int(res['valid'].sum())
        print(f'  Done in {elapsed:.1f}s — {n_valid_count}/{len(res)} combos '
              f'with >={min_trades} trades')

    return res


def save_results(res: pd.DataFrame, grid: dict, strategy_stem: str,
                 objective: str, run_dir: Path = None) -> Path:
    if run_dir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = RESULTS_DIR / f'{strategy_stem}_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(run_dir / 'results.csv', index=False)

    best = res[res['valid']].head(1)
    if not best.empty:
        best_params = {k: best.iloc[0][k] for k in grid if k in best.columns}
        best_params = {k: (v.item() if hasattr(v, 'item') else v)
                       for k, v in best_params.items()}
        meta = {
            'strategy': strategy_stem,
            'objective': objective,
            'best_params': best_params,
            'metrics': {c: (best.iloc[0][c].item() if hasattr(best.iloc[0][c], 'item')
                            else best.iloc[0][c])
                        for c in best.columns if c not in grid},
        }
        with open(run_dir / 'best_params.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)
    return run_dir
