"""
Combine the ASOS weather chunks into one file per year.

Input:  data/interim/asos/asos_<year>_<chunk>.parquet   (100 files)
Output: data/interim/asos/weather_<year>.parquet        (4 files)
"""

import pandas as pd
from pathlib import Path

ASOS = Path("data/interim/asos")

for year in [2023, 2024, 2025, 2026]:
    out = ASOS / f"weather_{year}.parquet"

    if out.exists():
        print("skip ", out.name)
        continue

    files = sorted(ASOS.glob(f"asos_{year}_*.parquet"))
    print(f"combining {len(files)} files for {year} ...")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # late December appears twice because each year was fetched 3 days early
    before = len(df)
    df = df.drop_duplicates(subset=["station", "valid"])
    print(f"  dropped {before - len(df):,} duplicates")

    # merge_asof needs this sorted, and sorting 35M rows is slow — do it once
    df = df.sort_values("valid")

    df.to_parquet(out, index=False)
    print(f"  wrote {out.name}  {len(df):,} rows, {df.station.nunique()} stations")