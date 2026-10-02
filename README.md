# Early Prediction of US Flight Departure Delays

CS 486-01 capstone. Anup Rijal, Truman State University.
Supervisor: Dr. Kafi Rahman.

Measures how flight delay prediction quality changes with how far ahead the
prediction is made — 24, 12, 6 and 2 hours before scheduled departure — using
only information that genuinely existed at each of those moments.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Pipeline

Run from the project root, in order:

| Script | Does | Output |
|---|---|---|
| `download_bts.py` | Fetch BTS On-Time zips, Jan 2023 – Jul 2026 | `data/raw/bts/` |
| `prepare_bts.py` | Trim 110 columns to 34 | `data/interim/` |
| `download_asos.py` | Fetch ASOS weather for all airports in the flight data | `data/raw/asos/` |
| `prepare_asos.py` | Convert to parquet, parse timestamps | `data/interim/asos/` |

Every script skips work already done, so they are safe to rerun after an
interruption.

## Data sources

- Flights: Bureau of Transportation Statistics, Reporting Carrier On-Time
  Performance — https://www.transtats.bts.gov/ONTIME/
- Weather: Iowa Environmental Mesonet ASOS archive —
  https://mesonet.agron.iastate.edu/request/download.phtml
- Airport metadata: `airportsdata` — https://pypi.org/project/airportsdata/

## Note on data

The `data/` folder is not in this repository — roughly 15 GB. Run the download
scripts to rebuild it.

See `docs/data_dictionary.md` for column meanings and known data issues.