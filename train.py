"""
Trains one XGBoost model per (horizon, feature set) and saves test predictions.

    python train.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from xgboost import XGBClassifier

from features import load

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
(OUT / "preds").mkdir(exist_ok=True)

TRAIN = [2023, 2024]
VAL   = [2025]
TEST  = [2026]

SAMPLE = None      # set to None for the final full run

rows = []
for h in [2, 6, 12, 24]:
    for fs in ["A", "B", "C"]:
        tag = f"h{h:02d}_{fs}"
        print(f"\n--- {tag} ---")

        Xtr, ytr = load(h, TRAIN, fs, sample=SAMPLE)
        Xva, yva = load(h, VAL,   fs, sample=SAMPLE)
        Xte, yte = load(h, TEST,  fs)
        print(f"train {len(Xtr):,}  val {len(Xva):,}  test {len(Xte):,}")

        model = XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=2000,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            eval_metric="auc",
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=200)

        p = model.predict_proba(Xte)[:, 1]

        row = {"horizon": h, "features": fs,
               "trees": model.best_iteration,
               "auc": roc_auc_score(yte, p),
               "pr_auc": average_precision_score(yte, p),
               "brier": brier_score_loss(yte, p),
               "base_rate": float(yte.mean())}
        rows.append(row)
        print(row)

        model.save_model(OUT / f"model_{tag}.json")
        np.save(OUT / "preds" / f"p_{tag}.npy", p)
        np.save(OUT / "preds" / f"y_{tag}.npy", yte.to_numpy())

results = pd.DataFrame(rows)
results.to_csv(OUT / "results.csv", index=False)
print("\n", results.to_string(index=False))