"""
Write a 20-row preview of each processed file, for eyeballing.

Input:  data/processed/*.parquet
Output: data/processed_header/header_<name>.csv
"""

import pandas as pd
from pathlib import Path

SRC = Path("data/processed")
OUT = Path("data/processed_header")
OUT.mkdir(parents=True, exist_ok=True)

for f in sorted(SRC.glob("*.parquet")):
    d = pd.read_parquet(f)

    # skip the unmatched rows — the files are sorted by prediction time, so a
    # plain head() gives Guam and Puerto Rico flights with empty weather
    preview = d[d.o_valid.notna()].head(20)

    out = OUT / f"header_{f.stem}.csv"
    preview.to_csv(out, index=False)
    print("wrote", out.name)