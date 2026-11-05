"""
LightGBM and CatBoost against XGBoost, feature set C.
Same 2,000,000-row sample for every model so the comparison is fair.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

from features import load

OUT = Path("outputs"); (OUT / "preds").mkdir(parents=True, exist_ok=True)

TRAIN, VAL, TEST = [2023, 2024], [2025], [2026]
SAMPLE = None
FS = "C"


def score(name, h, yte, p):
    row = {"model": name, "horizon": h,
           "auc": roc_auc_score(yte, p),
           "pr_auc": average_precision_score(yte, p),
           "brier": brier_score_loss(yte, p)}
    print(row)
    return row


rows = []
for h in [2, 6, 12, 24]:
    print(f"\n===== horizon {h}h =====")
    Xtr, ytr = load(h, TRAIN, FS, sample=SAMPLE)
    Xva, yva = load(h, VAL,   FS, sample=SAMPLE)
    Xte, yte = load(h, TEST,  FS)

    m = LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=255,
                       min_child_samples=50, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[early_stopping(50, verbose=False)])
    p = m.predict_proba(Xte)[:, 1]
    rows.append(score("LightGBM", h, yte, p))
    np.save(OUT / "preds" / f"p_lgb_h{h:02d}_C.npy", p)

    cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]

    def prep(X):
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype("object").fillna("NA").astype(str)
        return X

    m = CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=8,
                           eval_metric="AUC", cat_features=cats,
                           early_stopping_rounds=50, verbose=500,
                           thread_count=-1)
    m.fit(prep(Xtr), ytr, eval_set=(prep(Xva), yva))
    p = m.predict_proba(prep(Xte))[:, 1]
    rows.append(score("CatBoost", h, yte, p))
    np.save(OUT / "preds" / f"p_cat_h{h:02d}_C.npy", p)

df = pd.DataFrame(rows)
df.to_csv(OUT / "model_comparison_full.csv", index=False)
print("\n", df.to_string(index=False))