"""
Now this file will run through all the zip within raw/bts, 
and extract the csv files into data/interim/*.csv
"""


import zipfile 
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/bts")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

KEEP = [ #SCHEDULE: known when the flight is planned 
         #only legeal features 
    "FlightDate", "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "Reporting_Airline", "DOT_ID_Reporting_Airline",
    "Tail_Number", "Flight_Number_Reporting_Airline",
    "Origin", "Dest", "OriginState", "DestState",
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
    "DepTimeBlk", "ArrTimeBlk", "Distance", "DistanceGroup",

    #OUTCOMES: our prediction variables 
    "DepDelay", "DepDelayMinutes", "DepDel15",
    "ArrDelay", "ArrDel15",
    "Cancelled", "CancellationCode", "Diverted",
         
    #DELAY CAUSE: not features but maybe used later to check aircraft histroy 
    "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",   
    ]

#RAW.glob finds every zip in raw data inside bts 
#sorting puts them in order as per month. 
for zip_path in sorted(RAW.glob("*.zip")):
    # .stem is the filename without its extension: 
    # bts_2023_01.zip -> bts_2023_01
    # then + ".csv" and then on interim folder we get #data/interim/bts_2023_01.csv and so on....
    out = INTERIM / (zip_path.stem + ".csv")

    #CHECK if the csv is already made, if yes skip 
    if out.exists():
        print("skip ", out.name)
        continue

    #Open the zip file and extract the csv file inside it.
    with zipfile.ZipFile(zip_path) as z: 
        #Each zip file has one csv and one readme, we will only pick a csv file in it. 
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]

        #z.open() gives a file like object pandas can read it direclty. 
        #Using usecols filter at read time so other 76 columns never load. 

        with z.open(csv_name) as f: 
            df = pd.read_csv(f, usecols=lambda c: c in KEEP, low_memory= False) 

    df.to_csv(out, index=False)
    print("wrote", out.name, f"{len(df):,} rows", f"{df.shape[1]} cols")