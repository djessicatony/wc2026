"""Generate README visualizations into assets/."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "figure.dpi": 130})

# ── 1. Prediction vs market for Brazil–Morocco ──────────────────────────
outcomes = ["Brazil win", "Draw", "Morocco win"]
ours = [36, 28, 36]
statsbomb = [39, 32, 29]   # independent StatsBomb + XGBoost model (community)
market = [59, 26, 17]
x = np.arange(len(outcomes))
w = 0.27

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x - w, ours, w, label="Our model (logreg + Elo)", color="#2a9d8f")
ax.bar(x, statsbomb, w, label="StatsBomb + XGBoost (community)", color="#457b9d")
ax.bar(x + w, market, w, label="Polymarket", color="#e76f51")
ax.axvspan(0.5, 1.5, color="gold", alpha=0.15)
ax.text(1, 63, "ACTUAL: 1–1 draw", ha="center", fontweight="bold", color="#b8860b")
ax.set_xticks(x); ax.set_xticklabels(outcomes)
ax.set_ylabel("Probability (%)"); ax.set_ylim(0, 70)
ax.set_title("Two independent data models saw an even match — the market was the outlier")
ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
for i, (o, s, m) in enumerate(zip(ours, statsbomb, market)):
    ax.text(i - w, o + 1, f"{o}%", ha="center", fontsize=8)
    ax.text(i, s + 1, f"{s}%", ha="center", fontsize=8)
    ax.text(i + w, m + 1, f"{m}%", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig("assets/prediction_vs_market.png"); plt.close(fig)

# ── 2. Accuracy by version (the feature-engineering story) ──────────────
versions = ["v1\nform", "v2\nXGBoost\n(rich, few)", "v3\n+ Elo", "v6\nXGBoost\n(+Elo)", "v7\n+ importance"]
acc = [65.0, 57.3, 71.2, 70.7, 71.1]
colors = ["#264653", "#a44", "#2a9d8f", "#264653", "#264653"]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(versions, acc, color=colors)
ax.axhline(65, ls="--", color="grey", lw=1)
ax.set_ylabel("Backtest accuracy (%)"); ax.set_ylim(50, 75)
ax.set_title("One feature (Elo) beat switching models — features > models")
ax.spines[["top", "right"]].set_visible(False)
for b, a in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.4, f"{a:.1f}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig("assets/accuracy_by_version.png"); plt.close(fig)

print("saved assets/prediction_vs_market.png and assets/accuracy_by_version.png")
