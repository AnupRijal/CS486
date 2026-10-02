"""
Downloading flight data from Beaureau of Transportation Statistics (BTS) for the years 2023
Using loops to chanegt the year and month in the URL to download all the files.
"""
import requests 
from pathlib import Path

RAW= Path("data/raw/bts")

URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")

# month 1 through 12 
# Starting with 2023 
months = ([(2023, m) for m in range (1,13)] + 
            [(2024, m) for m in range (1,13)] + 
            [(2025, m) for m in range (1,13)] + 
            [(2026, m) for m in range (1,8)]
            )

for y, m in months:
    #padding 0 on file month so that files gets sorted correctly in future if needed on finder 
    out = RAW / f"bts_{y}_{m:02d}.zip"

    if out.exists(): 
        print(f"File {out} already exists, skipping download.")
        continue 

    print(f"get {y}-{m:02d} data from BTS")

    #There is no default timeout. so, adding stalled error 
    r = requests.get(URL.format(y=y, m=m), timeout=10)

    #stop on bad response. throw error if anything fails 
    r.raise_for_status()

    #ZIP is binary. Writing to file inbytes
    out.write_bytes(r.content)

    print("saved", out.name)