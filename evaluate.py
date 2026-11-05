"""
Reads outputs/results.csv and the saved predictions, prints the paper table
and draws the degradation figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs")
FIGS = OUT / "figs"; FIGS.mkdir(exist_ok=True)

r = pd.read_csv(OUT / "results.csv")

acc = []
for _, row in r.iterrows():
    tag = f"h{int(row.horizon):02d}_{row.features}"
    p = np.load(OUT / "preds" / f"p_{tag}.npy")
    y = np.load(OUT / "preds" / f"y_{tag}.npy")
    acc.append(((p >= 0.5) == y).mean())
r["accuracy"] = acc
r["always_on_time"] = 1 - r.base_rate

print(r[["horizon", "features", "auc", "pr_auc", "brier",
         "accuracy", "always_on_time"]].round(4).to_string(index=False))

w = r.pivot(index="horizon", columns="features", values="auc")
w["weather_gain"] = w.B - w.A
w["aircraft_gain"] = w.C - w.B
w["total_gain"] = w.C - w.A
print("\n", w.round(4).to_string())
w.to_csv(OUT / "gains.csv")

# how good is the model at picking out the worst flights
print(f"\n{'horizon':>8}  {'top 10% delayed':>16}  {'base rate':>10}  {'lift':>6}")
for h in [2, 6, 12, 24]:
    p = np.load(OUT / "preds" / f"p_h{h:02d}_C.npy")
    y = np.load(OUT / "preds" / f"y_h{h:02d}_C.npy")
    k = int(0.10 * len(p))
    top = np.argsort(p)[-k:]
    print(f"{h:>6}h  {y[top].mean():>16.1%}  {y.mean():>10.1%}  "
          f"{y[top].mean() / y.mean():>5.2f}x")

for h in [2, 6, 12, 24]:
    p = np.load(OUT / "preds" / f"p_h{h:02d}_C.npy")
    y = np.load(OUT / "preds" / f"y_h{h:02d}_C.npy")
    late = y.sum()

    flag = p >= 0.5
    print(f"\n{h}h  at threshold 0.5: flagged {flag.mean():.1%} of flights, "
          f"caught {y[flag].sum()/late:.1%} of delays, "
          f"{y[flag].mean():.1%} of flags correct")

    order = np.argsort(p)[::-1]
    for frac in [0.10, 0.20, 0.30]:
        top = order[:int(frac * len(p))]
        print(f"     flag top {frac:.0%}: caught {y[top].sum()/late:>5.1%} "
              f"of delays, {y[top].mean():>5.1%} of flags correct")

# AUC against horizon
fig, ax = plt.subplots(figsize=(6, 4))
style = {"A": ("Schedule only",      "#555555", "o", "--"),
         "B": ("+ weather",          "#3b7ea1", "s", "-"),
         "C": ("+ aircraft history", "#b4553a", "^", "-")}
for fs, (label, color, marker, ls) in style.items():
    sub = r[r.features == fs].sort_values("horizon")
    ax.plot(sub.horizon, sub.auc, ls, color=color, marker=marker,
            markersize=5, linewidth=1.5, label=label)
ax.set_xlabel("Hours before scheduled departure")
ax.set_ylabel("AUC")
ax.set_xticks([2, 6, 12, 24])
ax.invert_xaxis()
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGS / "auc_by_horizon.png", dpi=300)

# where the gain comes from
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(w))
ax.bar(x, w.weather_gain, 0.45, label="Weather", color="#3b7ea1")
ax.bar(x, w.aircraft_gain, 0.45, bottom=w.weather_gain,
       label="Aircraft history", color="#b4553a")
ax.set_xticks(x)
ax.set_xticklabels([f"{h}h" for h in w.index])
ax.set_xlabel("Hours before scheduled departure")
ax.set_ylabel("AUC gain over schedule only")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGS / "gain_sources.png", dpi=300)

print("\nwrote figures to", FIGS)