"""
Add UTC timestamps to the flight data.

BTS records scheduled departure as an hhmm integer in LOCAL time. ASOS records
weather in UTC. This converts the flights so the two can be compared.

Input:  data/interim/bts_<year>_<month>.parquet
Output: data/interim/flights_utc_<year>.parquet
"""

import airportsdata
import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
AIRPORTS = airportsdata.load("IATA")

for year in [2023, 2024, 2025, 2026]:
    out = INTERIM / f"flights_utc_{year}.parquet"

    if out.exists():
        print("skip ", out.name)
        continue

    files = sorted(INTERIM.glob(f"bts_{year}_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"{year}: {len(df):,} flights")

    # BTS writes midnight as 2400, which is not a real time
    hhmm = df.CRSDepTime.astype("Int64").fillna(0).astype(int).replace(2400, 0)

    naive = (pd.to_datetime(df.FlightDate)
             + pd.to_timedelta(hhmm // 100, unit="h")
             + pd.to_timedelta(hhmm % 100, unit="m"))

    # Each airport has its own timezone, so convert one airport at a time.
    # ambiguous and nonexistent handle the two clock changes each year.
    parts = []
    for code, group in naive.groupby(df.Origin):
        rec = AIRPORTS.get(code)
        if rec is None or not rec.get("tz"):
            continue
        parts.append(
            group.dt.tz_localize(rec["tz"], ambiguous=True,
                                 nonexistent="shift_forward").dt.tz_convert("UTC")
        )

    df["sched_dep_utc"] = pd.concat(parts).sort_index()

    before = len(df)
    df = df.dropna(subset=["sched_dep_utc"]).sort_values("sched_dep_utc")
    print(f"  dropped {before - len(df):,} with no timezone")

    df.to_parquet(out, index=False)
    print(f"  wrote {out.name}  {len(df):,} rows")