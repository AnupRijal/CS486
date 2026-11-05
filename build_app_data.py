"""
Builds the table the web app reads: one row per flight, with the model's
prediction at each horizon and what actually happened.
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("outputs")
FEATURES = Path("data/features")
YEAR = 2026
SAMPLE = 300_000

KEYS = ["FlightDate", "Reporting_Airline",
        "Flight_Number_Reporting_Airline", "Origin", "Dest"]

EXTRA = ["DepDel15", "DepDelay", "CRSDepTime", "o_vsby", "prev_dep_delay",
         "Distance", "CRSElapsedTime", "o_sknt", "o_wxcodes"]

base = None
for h in [2, 6, 12, 24]:
    df = pd.read_parquet(
        FEATURES / f"feat_h{h:02d}_{YEAR}.parquet",
        columns=KEYS + ["o_station"] + EXTRA)
    df = df[df.o_station.notna() & df.DepDel15.notna()].reset_index(drop=True)

    p = np.load(OUT / "preds" / f"p_h{h:02d}_C.npy")
    assert len(df) == len(p), f"{h}h: {len(df):,} rows but {len(p):,} predictions"
    df[f"p{h:02d}"] = p
    df = df.drop_duplicates(subset=KEYS)

    keep = KEYS + [f"p{h:02d}"]
    if base is None:
        base = df[keep + EXTRA]
    else:
        base = base.merge(df[keep], on=KEYS, how="inner")
    print(f"{h:>2}h -> {len(base):,} flights")

if len(base) > SAMPLE:
    base = base.sample(SAMPLE, random_state=0)

base.sort_values(KEYS).reset_index(drop=True).to_parquet(
    OUT / "app_data.parquet", index=False)
print(f"\nwrote app_data.parquet  {len(base):,} flights")