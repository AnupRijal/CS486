import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
YEAR = 2023

weather = pd.read_parquet(INTERIM / "asos" / f"weather_{YEAR}.parquet",
                         columns=["station", "valid", "vsby"]).dropna(subset=["vsby"])

print(f"{'horizon':>8}  {'still low at departure':>24}  {'n':>10}")

for h in [2, 6, 12, 24]:
    df = pd.read_parquet(PROCESSED / f"merged_h{h:02d}_{YEAR}.parquet",
                         columns=["Origin", "sched_dep_utc", "o_vsby"])
    df = df[df.o_vsby < 3].sort_values("sched_dep_utc")

    m = pd.merge_asof(df, weather,
                      left_on="sched_dep_utc", right_on="valid",
                      left_by="Origin", right_by="station",
                      direction="backward", tolerance=pd.Timedelta("90min"))
    m = m.dropna(subset=["vsby"])

    print(f"{h:>6}h  {(m.vsby < 3).mean():>23.1%}  {len(m):>10,}")

v = pd.read_parquet(PROCESSED / f"merged_h06_{YEAR}.parquet", columns=["o_vsby"]).dropna()
print(f"\nbase rate of low visibility: {(v.o_vsby < 3).mean():.1%}")