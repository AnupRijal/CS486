"""
Fetch ASOS weather for every airport in the BTS data.
Output: data/raw/asos/asos_<year>_<chunk>.csv
"""

import time
import pandas as pd
import requests
from pathlib import Path

INTERIM = Path("data/interim")
ASOS = Path("data/raw/asos")
ASOS.mkdir(parents=True, exist_ok=True)

# Airport list, busiest first, straight from the flight data
files = sorted(INTERIM.glob("bts_*.csv"))
counts = pd.concat([pd.read_csv(f, usecols=["Origin"]) for f in files]).Origin.value_counts()
airports = list(counts.index)
print(len(airports), "airports")

WX = ["tmpf", "dwpf", "relh", "sknt", "drct", "gust",
      "vsby", "p01i", "alti", "skyc1", "skyl1", "wxcodes"]

# Each year starts 3 days early so a T-24h prediction on Jan 1
# still has weather available before it.
RANGES = {
    2023: ("2022-12-29", "2024-01-01"),
    2024: ("2023-12-29", "2025-01-01"),
    2025: ("2024-12-29", "2026-01-01"),
    2026: ("2025-12-29", "2026-08-01"),
}

CHUNK = 15   # smaller chunks = smaller transfers = fewer dropped connections
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

for year, (start, end) in RANGES.items():
    for i in range(0, len(airports), CHUNK):
        batch = airports[i:i + CHUNK]
        out = ASOS / f"asos_{year}_{i // CHUNK:02d}.csv"

        if out.exists():
            print("skip ", out.name)
            continue

        params = (
            [("station", a) for a in batch]
            + [("data", c) for c in WX]
            + [("sts", f"{start}T00:00:00Z"),
               ("ets", f"{end}T00:00:00Z"),
               ("tz", "UTC"),
               ("format", "onlycomma"),
               ("missing", "empty"),
               ("trace", "0.0001"),
               ("direct", "yes")]
        )

        print(f"get   {out.name}  ({len(batch)} stations)")

        # IEM drops large transfers sometimes. Try three times before giving up.
        for attempt in range(3):
            try:
                r = requests.get(URL, params=params, timeout=1800)
                r.raise_for_status()
                break
            except Exception as e:
                print(f"      attempt {attempt + 1} failed: {type(e).__name__}")
                if attempt == 2:
                    raise
                time.sleep(15)

        out.write_text(r.text)
        print(f"saved {out.name}  {len(r.text)/1e6:.1f} MB")