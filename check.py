import pandas as pd

d = pd.read_parquet("data/interim/flights_utc_2023.parquet",
                    columns=["FlightDate", "Origin", "CRSDepTime", "sched_dep_utc"])

ord_ = d[d.Origin == "ORD"].copy()
ord_["offset_hours"] = (ord_.sched_dep_utc.dt.hour - ord_.CRSDepTime // 100) % 24

print(ord_.groupby(ord_.FlightDate.astype(str).str[:7]).offset_hours.median())