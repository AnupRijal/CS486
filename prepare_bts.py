"""
Read each BTS zip, keep only the columns we need, save as parquet.

Input:  data/raw/bts/*.zip
Output: data/interim/*.parquet
"""

import zipfile
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/bts")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

KEEP = [
    # schedule: known when the flight is planned (legal features)
    "FlightDate", "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "Reporting_Airline", "DOT_ID_Reporting_Airline",
    "Tail_Number", "Flight_Number_Reporting_Airline",
    "Origin", "Dest", "OriginState", "DestState",
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
    "DepTimeBlk", "ArrTimeBlk", "Distance", "DistanceGroup",

    # outcomes: what we are predicting
    "DepDelay", "DepDelayMinutes", "DepDel15",
    "ArrDelay", "ArrDel15",
    "Cancelled", "CancellationCode", "Diverted",

    # delay causes: not features, used later to check aircraft history work
    "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
]

for zip_path in sorted(RAW.glob("*.zip")):
    out = INTERIM / (zip_path.stem + ".parquet")

    if out.exists():
        print("skip  ", out.name)
        continue

    with zipfile.ZipFile(zip_path) as z:
        # each zip holds one csv plus a readme — pick the csv
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]

        # usecols filters at read time, so the other 76 columns never load
        with z.open(csv_name) as f:
            df = pd.read_csv(f, usecols=lambda c: c in KEEP, low_memory=False)

    df.to_parquet(out, index=False)
    print("wrote ", out.name, f"{len(df):,} rows", f"{df.shape[1]} cols")