import pandas as pd

df = pd.read_parquet("data/processed/merged_h06_2023.parquet")

print(f"{len(df):,} rows\n")
for c in df.columns:
    print(f"{c:35s} {df[c].notna().mean():6.1%}")