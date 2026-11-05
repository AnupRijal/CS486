import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
BLOCKS = ["0700-0759", "1300-1359", "1900-1959"]
HORIZONS = [2, 6, 12, 24]

kept = {}
for h in HORIZONS:
    parts = []
    for f in sorted(PROCESSED.glob(f"merged_h{h:02d}_*.parquet")):
        d = pd.read_parquet(f, columns=["o_vsby", "DepDel15", "DepTimeBlk"])
        parts.append(d[d.DepTimeBlk.isin(BLOCKS)])
    kept[h] = pd.concat(parts).dropna(subset=["o_vsby", "DepDel15"])

for block in BLOCKS:
    print(f"\ndepartures scheduled {block}")
    print(f"{'horizon':>8}  {'low vis':>8}  {'normal':>8}  {'gap':>8}  {'n':>10}")
    for h in HORIZONS:
        d = kept[h]
        d = d[d.DepTimeBlk == block]
        low = d.DepDel15[d.o_vsby < 3]
        normal = d.DepDel15[d.o_vsby >= 3]
        print(f"{h:>6}h  {low.mean():>8.1%}  {normal.mean():>8.1%}  "
              f"{(low.mean() - normal.mean()) * 100:>6.1f}pp  {len(low):>10,}")