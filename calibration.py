"""
Are the predicted probabilities trustworthy, and how sharp are they?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs")
PREDS = OUT / "preds"
FIGS = OUT / "figs"; FIGS.mkdir(exist_ok=True)

HORIZONS = [2, 6, 12, 24]
COLORS = {2: "#b4553a", 6: "#c08a3e", 12: "#4a7a8c", 24: "#555555"}
BINS = np.linspace(0, 1, 21)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
rows = []

for h in HORIZONS:
    p = np.load(PREDS / f"p_h{h:02d}_C.npy")
    y = np.load(PREDS / f"y_h{h:02d}_C.npy")

    idx = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    obs, pred, n = [], [], []
    for b in range(len(BINS) - 1):
        m = idx == b
        if m.sum() < 500:
            continue
        obs.append(y[m].mean()); pred.append(p[m].mean()); n.append(m.sum())

    obs, pred, n = np.array(obs), np.array(pred), np.array(n)
    rows.append({"horizon": h,
                 "ece": (n / n.sum() * np.abs(obs - pred)).sum(),
                 "pred_std": p.std(),
                 "p10": np.percentile(p, 10),
                 "p90": np.percentile(p, 90)})

    ax1.plot(pred, obs, "-o", color=COLORS[h], markersize=4,
             linewidth=1.3, label=f"{h}h")
    ax2.hist(p, bins=50, histtype="step", color=COLORS[h],
             linewidth=1.3, density=True, label=f"{h}h")

ax1.plot([0, 1], [0, 1], ":", color="#999999", linewidth=1)
ax1.set_xlabel("Predicted probability of delay")
ax1.set_ylabel("Observed delay rate")
ax1.set_title("Calibration")
ax1.legend(frameon=False)

ax2.set_xlabel("Predicted probability of delay")
ax2.set_ylabel("Density")
ax2.set_title("Sharpness")
ax2.legend(frameon=False)

for a in (ax1, ax2):
    a.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(FIGS / "calibration.png", dpi=300)

t = pd.DataFrame(rows)
t.to_csv(OUT / "calibration.csv", index=False)
print(t.round(4).to_string(index=False))