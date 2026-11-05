"""
Convert raw ASOS CSVs to parquet.
Input:  data/raw/asos/*.csv
Output: data/interim/asos/*.parquet
"""

import pandas as pd
from pathlib import Path

RAW = Path("data/raw/asos")
OUT = Path("data/interim/asos")
OUT.mkdir(parents=True, exist_ok=True)

for csv_path in sorted(RAW.glob("*.csv")):
    out = OUT / (csv_path.stem + ".parquet")

    if out.exists():
        print("skip ", out.name)
        continue

    # comment="#" skips the header lines IEM puts at the top
    df = pd.read_csv(csv_path, comment="#", low_memory=False)

    # Parse timestamps now so every later read gets them for free
    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")

    df.to_parquet(out, index=False)
    print("wrote", out.name, f"{len(df):,} rows")