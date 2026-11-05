import pandas as pd
from pathlib import Path

df = pd.read_parquet(Path("data/interim/asos") / "asos_2023_00.parquet")
df["valid"] = pd.to_datetime(df.valid)

print(f"{len(df):,} rows, {df.station.nunique()} stations\n")

one = df[df.station == df.station.iloc[0]].sort_values("valid")
print(f"station {one.station.iloc[0]}, first 30 observations")
print(one[["valid", "tmpf", "sknt", "vsby", "alti"]].head(30).to_string(index=False))

print("\nminutes between observations:")
print(one.valid.diff().dt.total_seconds().div(60).value_counts().head(10).to_string())