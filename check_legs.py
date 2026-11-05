import pandas as pd
from pathlib import Path

df = pd.read_parquet(Path("data/processed") / "merged_h02_2023.parquet",
                     columns=["Tail_Number", "Origin", "Dest", "sched_dep_utc",
                              "CRSElapsedTime", "DepDelay", "ArrDelay"])

df["actual_dep_utc"] = df.sched_dep_utc + pd.to_timedelta(df.DepDelay, unit="m")
df["actual_arr_utc"] = (df.sched_dep_utc
                        + pd.to_timedelta(df.CRSElapsedTime, unit="m")
                        + pd.to_timedelta(df.ArrDelay, unit="m"))

one = df[df.Tail_Number == "N594JB"].sort_values("sched_dep_utc")
print(one[["Origin", "Dest", "sched_dep_utc", "actual_dep_utc",
           "actual_arr_utc", "ArrDelay"]].head(12).to_string(index=False))