"""
Records every value each categorical column can take, so train, validation
and test all encode them identically. Run once.
"""

import json
import pandas as pd
from pathlib import Path

FEATURES = Path("data/features")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

COLS = ["Reporting_Airline", "Origin", "Dest", "o_skyc1"]

seen = {c: set() for c in COLS}
for f in sorted(FEATURES.glob("feat_h06_*.parquet")):
    df = pd.read_parquet(f, columns=COLS)
    for c in COLS:
        seen[c].update(df[c].dropna().unique())
    print("read", f.name)

cats = {c: sorted(v) for c, v in seen.items()}
(OUT / "categories.json").write_text(json.dumps(cats, indent=2))

for c, v in cats.items():
    print(f"{c:20s} {len(v)}")