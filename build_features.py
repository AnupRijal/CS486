"""
Adds aircraft-history features, as known at the prediction moment:
  prev_arr_*  the last leg this aircraft had LANDED from
  prev_dep_*  the last leg it had TAKEN OFF on (may still be in the air)
"""

import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
FEATURES = Path("data/features")
FEATURES.mkdir(parents=True, exist_ok=True)

for year in [2023, 2024, 2025, 2026]:
    for h in [2, 6, 12, 24]:
        out = FEATURES / f"feat_h{h:02d}_{year}.parquet"
        if out.exists():
            print("skip ", out.name); continue

        df = pd.read_parquet(PROCESSED / f"merged_h{h:02d}_{year}.parquet")

        df["actual_dep_utc"] = df.sched_dep_utc + pd.to_timedelta(df.DepDelay, unit="m")
        df["actual_arr_utc"] = (df.sched_dep_utc
                                + pd.to_timedelta(df.CRSElapsedTime, unit="m")
                                + pd.to_timedelta(df.ArrDelay, unit="m"))

        # legs this aircraft had already LANDED from
        legs = (df[["Tail_Number", "sched_dep_utc", "actual_arr_utc", "ArrDelay"]]
                .dropna(subset=["Tail_Number", "actual_arr_utc"])
                .rename(columns={"sched_dep_utc":  "prev_arr_sched",
                                 "actual_arr_utc": "prev_arr_utc",
                                 "ArrDelay":       "prev_arr_delay"})
                .sort_values("prev_arr_utc"))

        m = pd.merge_asof(df.sort_values("pred_time_utc"), legs,
                          left_on="pred_time_utc", right_on="prev_arr_utc",
                          by="Tail_Number", direction="backward",
                          tolerance=pd.Timedelta("48h"))

        bad = m.prev_arr_sched >= m.sched_dep_utc
        m.loc[bad, "prev_arr_delay"] = float("nan")
        m.loc[bad, "prev_arr_utc"] = pd.NaT

        # legs it had already TAKEN OFF on, landed or not
        deps = (df[["Tail_Number", "sched_dep_utc", "actual_dep_utc", "DepDelay"]]
                .dropna(subset=["Tail_Number", "actual_dep_utc"])
                .rename(columns={"sched_dep_utc":  "prev_dep_sched",
                                 "actual_dep_utc": "prev_dep_utc",
                                 "DepDelay":       "prev_dep_delay"})
                .sort_values("prev_dep_utc"))

        m = pd.merge_asof(m, deps,
                          left_on="pred_time_utc", right_on="prev_dep_utc",
                          by="Tail_Number", direction="backward",
                          tolerance=pd.Timedelta("48h"))

        bad = m.prev_dep_sched >= m.sched_dep_utc
        m.loc[bad, "prev_dep_delay"] = float("nan")
        m.loc[bad, "prev_dep_utc"] = pd.NaT

        m["hours_since_prev_arr"] = (
            (m.pred_time_utc - m.prev_arr_utc).dt.total_seconds() / 3600)
        m["hours_since_prev_dep"] = (
            (m.pred_time_utc - m.prev_dep_utc).dt.total_seconds() / 3600)

        # after-the-fact — never let these reach a model
        m = m.drop(columns=["actual_dep_utc", "actual_arr_utc"])

        m.to_parquet(out, index=False)
        print(f"wrote {out.name}  "
              f"arr {m.prev_arr_delay.notna().mean():.0%} "
              f"(med {m.hours_since_prev_arr.median():.1f}h)  "
              f"dep {m.prev_dep_delay.notna().mean():.0%} "
              f"(med {m.hours_since_prev_dep.median():.1f}h)")