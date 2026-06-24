# TMBB-VBT — Vectorbt Backtesting Engine for Futures Strategies

A fast, production-grade backtesting and walk-forward validation framework for NQ/ES futures strategies, built on [vectorbt](https://vectorbt.dev/).

**50–200× faster than event-driven backtesting** (backtrader/NT8) because signal generation is pure NumPy and portfolio simulation runs all parameter combinations simultaneously via vectorbt's vectorised engine.

---

## Features

- **Grid optimisation** — exhaustive parameter search with memory-bounded chunked processing and crash-resume checkpointing
- **Anchored walk-forward validation** — expanding IS windows; honest OOS test on unseen data
- **Automatic checkpointing** — interrupt and resume any grid run without losing work
- **Numba JIT** — optional acceleration on inner signal loops (falls back gracefully if not installed)
- **Strategy plugin contract** — add a new strategy in one file, `compute_signals(df, params) → dict`

---

## Strategies Included

### Consolidation Range with Signals (Zeiierman)
Port of the popular PineScript v6 indicator. Detects ranging conditions via ADX or volatility compression, then trades breakouts from a step-filter.

**Walk-forward results — MNQ 5min, 17.5 months, 4 OOS folds:**

| Fold | OOS Period | Trades | Win% | P&L | pnl_dd |
|------|-----------|--------|------|-----|--------|
| 1 | Apr–Jul 2025 | 151 | 45.7% | -$1,002 | FAIL |
| 2 | Jul–Nov 2025 | 193 | 77.2% | +$3,477 | PASS |
| 3 | Nov 25–Feb 26 | 177 | 74.0% | +$2,465 | PASS |
| 4 | Feb–Jun 2026 | 164 | 69.5% | +$332 | PASS |

**Total OOS: +$5,272 | 3/4 folds pass**

WF-optimal parameters: `ADX, len=14, band_mult=1.8, sl=0.7, tp=0.3, cooldown=10, adx_thresh=17`
> Note: `tp=0.3` is the key WF finding — phase-2 IS-optimum was 0.5 but all OOS folds consistently chose 0.3.

PineScript (TradingView) and NinjaTrader 8 NinjaScript ports in `strategies/`.

### IFVG Sniper
Inverted Fair Value Gap strategy. Detects Bull/Bear FVGs (3-bar price gaps), then enters when the FVG is later invalidated (price re-enters the gap). Quality filters (gap size, body ratio, range ATR) reduce noise.

---

## Project Structure

```
tmbb-vbt/
├── engine.py              # Core: data loading, VBT portfolio runner, metrics
├── optimize.py            # Grid optimizer with chunking + checkpointing
├── walkforward.py         # Anchored-expanding walk-forward validator
│
├── strategies/
│   ├── consolidation_range.py       # Consolidation Range signal generator
│   ├── ifvg_sniper.py               # IFVG Sniper signal generator
│   └── ConsolidationRange_WF.pine   # PineScript v6 (TradingView) — WF defaults
│
├── scripts/
│   └── make_tf_csvs.py    # Resample 1-min data to higher timeframes
│
├── _run_consol_range.py   # Phase 1 (default params) + Phase 2 (grid) for Consolidation Range
├── _wf_consol_range.py    # Walk-forward validation for Consolidation Range
│
└── results/
    └── consolidation_range_*_walkforward/
        └── wf_summary.csv   # OOS results per fold
```

---

## Installation

```bash
pip install vectorbt pandas numpy scipy
# Optional but recommended for ~10x signal loop speedup:
pip install numba
```

Python 3.10+ recommended. Tested on 3.12 and 3.14.

---

## Data Format

Place OHLCV CSV files in one of:
- `data/` subfolder next to `engine.py` (auto-detected)
- A sibling `TMBB_Backtest/data/` directory
- Any path set via `TMBB_DATA_DIR` environment variable

```
export TMBB_DATA_DIR=/path/to/your/data
```

CSV format — either:

**Headered** (any separator):
```
datetime,Open,High,Low,Close,Volume
2024-01-02 09:30:00,16800.0,16850.5,...
```

**NT8 raw** (semicolon-separated, no header):
```
20240102 093000;16800.00;16850.50;16780.00;16830.25;1234
```

The engine auto-detects both formats.

---

## Usage

### Single run (default params)
```python
import engine
from strategies import consolidation_range as cr

df = engine.load_csv(engine.DATA_DIR / 'MNQ_5min_stitched.csv')
metrics = engine.run_single(cr, df, cr.DEFAULT_PARAMS)
print(metrics)
```

### Grid optimisation
```bash
python _run_consol_range.py
```
Results are saved to `results/consolidation_range_<tf>/`. Checkpointing means you can kill and resume at any time.

### Walk-forward validation
```bash
python _wf_consol_range.py
```
Results are saved to `results/consolidation_range_<tf>_walkforward/wf_summary.csv`.

### Adding a new strategy
Create `strategies/my_strategy.py` with:

```python
DEFAULT_PARAMS = dict(...)
PARAM_GRID = {...}

def compute_signals(df, params):
    # return dict with: long_entries, short_entries, sl_frac, tp_frac, qty
    # all arrays of length len(df)
    # sl_frac / tp_frac: positive fraction of entry close price
    return dict(
        long_entries=...,
        short_entries=...,
        sl_frac=...,
        tp_frac=...,
        qty=...,
    )
```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| `pnl_dd` | P&L ÷ max drawdown (primary ranking metric) |
| `win%` | Percentage of trades that are profitable |
| `profit_factor` | Gross profit ÷ gross loss |
| `sharpe` | Annualised Sharpe ratio (via vectorbt) |
| `expectancy` | Average P&L per trade in dollars |

---

## Instrument Defaults

Configured for **MNQ** (Micro Nasdaq futures):
- Point value: $2.00
- Commission: $0.62/side
- Tick size: 0.25

Edit `POINT_VALUE`, `COMMISSION`, `TICK_SIZE` in `engine.py` for other instruments (e.g. MES: `$1.25/pt`, ES: `$12.50/pt`).

---

## Walk-Forward Methodology

**Anchored-expanding IS** — the IS window always starts from bar 0 and expands forward. Each OOS window is a fixed-size non-overlapping chunk at the front of the remaining data.

```
[─────────── IS ───────────][──OOS──]   Fold 4
[──────── IS ────────][──OOS──]         Fold 3
[───── IS ─────][──OOS──]               Fold 2
[── IS ──][──OOS──]                     Fold 1
```

This mirrors how a live trader would build confidence — more data strengthens the IS optimisation rather than sliding a fixed window.

**PASS criterion:** OOS P&L > 0. Strategy is considered deployable if ≥ 3/4 folds pass with consistent IS parameter selections.
