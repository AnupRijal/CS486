"""
The three feature sets, and a loader that returns them ready for XGBoost.

    from features import load
    X, y = load(horizon=6, years=[2023, 2024], feature_set="C")
"""

import json
import pandas as pd
from pathlib import Path

FEATURES = Path("data/features")
CATEGORIES = json.loads(Path("outputs/categories.json").read_text())

TARGET = "DepDel15"

# known as soon as the flight is scheduled - identical at every horizon
SCHEDULE = ["Month", "DayOfWeek", "dep_minutes",
            "Reporting_Airline", "Origin", "Dest",
            "Distance", "CRSElapsedTime"]

# observed at the prediction moment
WEATHER = ["o_vsby", "o_sknt", "o_gust", "o_p01i", "o_alti",
           "o_skyc1", "o_skyl1", "t_tmpf", "dew_spread",
           "wx_snow", "wx_rain", "wx_thunder", "wx_freezing"]

# what this aircraft had already done at the prediction moment
AIRCRAFT = ["prev_arr_delay", "hours_since_prev_arr",
            "prev_dep_delay", "hours_since_prev_dep"]

FEATURE_SETS = {"A": SCHEDULE,
                "B": SCHEDULE + WEATHER,
                "C": SCHEDULE + WEATHER + AIRCRAFT}

RAW = ["Month", "DayOfWeek", "CRSDepTime", "Reporting_Airline",
       "Origin", "Dest", "Distance", "CRSElapsedTime",
       "o_station", "o_vsby", "o_sknt", "o_gust", "o_p01i", "o_alti",
       "o_skyc1", "o_skyl1", "o_wxcodes", "t_tmpf", "t_dwpf",
       "prev_arr_delay", "hours_since_prev_arr",
       "prev_dep_delay", "hours_since_prev_dep", TARGET]


def load(horizon, years, feature_set, sample=None, seed=0):
    frames = []
    for y in years:
        df = pd.read_parquet(FEATURES / f"feat_h{horizon:02d}_{y}.parquet",
                             columns=RAW)
        df = df[df.o_station.notna() & df[TARGET].notna()]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if sample and sample < len(df):
        df = df.sample(sample, random_state=seed)

    # minutes past midnight, so 1359 -> 1400 is one step and not forty-one
    hhmm = df.CRSDepTime.astype("int32").replace(2400, 0)
    df["dep_minutes"] = (hhmm // 100) * 60 + (hhmm % 100)

    df["dew_spread"] = df.t_tmpf - df.t_dwpf

    wx = df.o_wxcodes.fillna("")
    df["wx_snow"]     = wx.str.contains("SN").astype("int8")
    df["wx_rain"]     = wx.str.contains("RA").astype("int8")
    df["wx_thunder"]  = wx.str.contains("TS").astype("int8")
    df["wx_freezing"] = wx.str.contains("FZ").astype("int8")

    X = df[FEATURE_SETS[feature_set]].copy()
    for c, levels in CATEGORIES.items():
        if c in X:
            X[c] = pd.Categorical(X[c], categories=levels)

    return X, df[TARGET].astype("int8")