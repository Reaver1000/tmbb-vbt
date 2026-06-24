"""Pre-resample MNQ 1-min stitched data into dedicated timeframe CSVs.

Output files land in TMBB_Backtest/data/ alongside the source file so
engine.list_data_files() picks them up automatically.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import sys as _sys
_sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import engine

TIMEFRAMES = ['3min', '5min', '15min', '30min', '60min', '4h']

SOURCES = {
    'MNQ': 'MNQ_1min_stitched.csv',
    'MES': 'MES_1min_stitched.csv',
}

for instrument, src_name in SOURCES.items():
    src = engine.DATA_DIR / src_name
    if not src.exists():
        print(f'  SKIP {src_name} (not found)')
        continue
    print(f'Loading {src_name}...')
    df1 = engine.load_csv(src)
    print(f'  {len(df1):,} bars  {df1.index[0].date()} to {df1.index[-1].date()}')
    for freq in TIMEFRAMES:
        fname = f'{instrument}_{freq}_stitched.csv'
        out = engine.DATA_DIR / fname
        df = engine.resample(df1, freq)
        df.to_csv(out)
        print(f'  {freq:>5}  {len(df):>7,} bars  ->  {fname}')
    print()

print('Done.')
