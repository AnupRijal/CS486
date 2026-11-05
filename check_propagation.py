import pandas as pd
from pathlib import Path

FEATURES = Path("data/features")

print(f"{'horizon':>8}  {'prev late':>10}  {'prev on time':>13}  {'gap':>8}  "
      f"{'median hrs since':>17}")

for h in [2, 6, 12, 24]:
    df = pd.read_parquet(FEATURES / f"feat_h{h:02d}_2023.parquet",
                         columns=["prev_arr_delay", "hours_since_prev_arr",
                                  "DepDel15"]).dropna()

    late = df.DepDel15[df.prev_arr_delay >= 15]
    ontime = df.DepDel15[df.prev_arr_delay < 15]

    print(f"{h:>6}h  {late.mean():>10.1%}  {ontime.mean():>13.1%}  "
          f"{(late.mean() - ontime.mean()) * 100:>6.1f}pp  "
          f"{df.hours_since_prev_arr.median():>16.1f}")