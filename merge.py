"""
Attaches origin-airport weather to each flight as it existed at the moment the
prediction is made, for four prediction horizons.

Weather arrives on two clocks: wind, visibility and pressure every 5 minutes,
temperature on the hourly cycle. Each group is joined on its own clock so every
value stays a real observation with a real timestamp.
"""

import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

HORIZONS = [24, 12, 6, 2]

FAST = ["sknt", "drct", "gust", "vsby", "p01i", "alti", "skyc1", "skyl1", "wxcodes"]
SLOW = ["tmpf", "dwpf", "relh"]

for year in [2023, 2024, 2025, 2026]:
    print(year)
    flights = pd.read_parquet(INTERIM / f"flights_utc_{year}.parquet")
    weather = pd.read_parquet(INTERIM / "asos" / f"weather_{year}.parquet")

    fast = weather[["station", "valid"] + FAST].add_prefix("o_")
    # keep only rows where the station actually reported temperature
    slow = weather[["station", "valid"] + SLOW].dropna(subset=["tmpf"]).add_prefix("t_")

    for h in HORIZONS:
        out = PROCESSED / f"merged_h{h:02d}_{year}.parquet"
        if out.exists():
            print(f"  skip  {out.name}"); continue

        f = flights.copy()
        f["pred_time_utc"] = f.sched_dep_utc - pd.Timedelta(hours=h)
        f = f.sort_values("pred_time_utc")

        m = pd.merge_asof(f, fast,
                          left_on="pred_time_utc", right_on="o_valid",
                          left_by="Origin", right_by="o_station",
                          direction="backward", tolerance=pd.Timedelta("90min"))

        m = pd.merge_asof(m, slow,
                          left_on="pred_time_utc", right_on="t_valid",
                          left_by="Origin", right_by="t_station",
                          direction="backward", tolerance=pd.Timedelta("2h"))

        m.to_parquet(out, index=False)
        print(f"  wrote {out.name}  {len(m):,} rows  "
              f"vsby {m.o_vsby.notna().mean():.1%}  tmpf {m.t_tmpf.notna().mean():.1%}")