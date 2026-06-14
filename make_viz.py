"""Generate README visualizations into assets/ (seaborn-styled)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

TEAL, BLUE, ORANGE = "#2a9d8f", "#3d5a80", "#ee6c4d"


def grouped(outcomes, series, title, savename, ymax, actual=None, actual_idx=None):
    """3-series grouped bars with value labels."""
    x = np.arange(len(outcomes)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, (label, vals, color) in enumerate(series):
        bars = ax.bar(x + (k - 1) * w, vals, w, label=label, color=color,
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.012, f"{v}%",
                    ha="center", fontsize=10, color="#333")
    if actual is not None:
        ax.axvspan(actual_idx - 0.5, actual_idx + 0.5, color="#ffd166", alpha=0.18, zorder=0)
        ax.text(actual_idx, ymax * 0.9, actual, ha="center", fontweight="bold", color="#b8860b")
    ax.set_xticks(x); ax.set_xticklabels(outcomes)
    ax.set_ylabel("Probability (%)"); ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9)
    sns.despine()
    fig.savefig(f"assets/{savename}.png"); plt.close(fig)


# 1. Brazil vs Morocco
grouped(
    ["Brazil win", "Draw", "Morocco win"],
    [("Our model (logreg + Elo)", [36, 28, 36], TEAL),
     ("sujar.tech (StatsBomb + XGBoost)", [39, 32, 29], BLUE),
     ("Polymarket", [59, 26, 17], ORANGE)],
    "Brazil vs Morocco — two data models saw an even match; the market didn't",
    "prediction_vs_market", ymax=70, actual="ACTUAL: 1–1 draw", actual_idx=1)

# 1b. Netherlands vs Japan
grouped(
    ["Netherlands", "Draw", "Japan"],
    [("Our model (logreg + Elo)", [36, 30, 34], TEAL),
     ("sujar.tech (StatsBomb + XGBoost)", [53, 29, 18], BLUE),
     ("Polymarket", [48, 28, 26], ORANGE)],
    "Netherlands vs Japan — our model called it even; it finished 2–2",
    "prediction_netherlands_japan", ymax=60, actual="ACTUAL: 2–2 draw", actual_idx=1)

# 2. Accuracy by version
versions = ["v1\nform", "v2\nXGBoost\n(rich, few)", "v3\n+ Elo", "v6\nXGBoost\n(+Elo)", "v7\n+ importance"]
acc = [65.0, 57.3, 71.2, 70.7, 71.1]
colors = ["#3d5a80", "#bc4749", TEAL, "#3d5a80", "#3d5a80"]
fig, ax = plt.subplots(figsize=(9, 4.8))
bars = ax.bar(versions, acc, color=colors, edgecolor="white", linewidth=0.6)
ax.axhline(65, ls="--", color="grey", lw=1, alpha=0.7)
ax.set_ylabel("Backtest accuracy (%)"); ax.set_ylim(50, 75)
ax.set_title("One feature (Elo) beat switching models — features > models",
             fontsize=13, fontweight="bold", pad=12)
for b, a in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.4, f"{a:.1f}", ha="center", fontsize=10, color="#333")
sns.despine()
fig.savefig("assets/accuracy_by_version.png"); plt.close(fig)

print("saved 4 charts to assets/")

# ── 3. Reliability curve (model calibration on the test set) ────────────
from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"]).dropna(subset=["home_score"]).sort_values("date")
r = defaultdict(lambda: 1500.0); rows = []
for _, m in raw.iterrows():
    h, a = m.home_team, m.away_team; rh, ra = r[h], r[a]
    rows.append({"date": m.date, "home_team": h, "away_team": a, "home_elo": rh, "away_elo": ra})
    e = 1/(1+10**((ra-rh)/400)); s = 1.0 if m.home_score>m.away_score else (0.5 if m.home_score==m.away_score else 0.0)
    r[h]=rh+30*(s-e); r[a]=ra+30*((1-s)-(1-e))
elo = pd.DataFrame(rows)
df = pd.read_csv("data/training_set.csv", parse_dates=["date"]).merge(elo, on=["date","home_team","away_team"]).dropna().sort_values("date")
F = ["home_win_rate","home_gf","home_ga","away_win_rate","away_gf","away_ga","is_neutral","home_elo","away_elo"]
cut = int(len(df)*0.8); tr, te = df.iloc[:cut], df.iloc[cut:]

raw_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[F], tr["home_won"])
cal_model = CalibratedClassifierCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), method="isotonic", cv=3).fit(tr[F], tr["home_won"])
p_raw = raw_model.predict_proba(te[F])[:,1]
p_cal = cal_model.predict_proba(te[F])[:,1]
yt_raw, xp_raw = calibration_curve(te["home_won"], p_raw, n_bins=10)
yt_cal, xp_cal = calibration_curve(te["home_won"], p_cal, n_bins=10)

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot([0,1],[0,1], "--", color="grey", label="Perfect")
ax.plot(xp_raw, yt_raw, "o-", color="#8d99ae", label="Raw", markersize=6)
ax.plot(xp_cal, yt_cal, "o-", color="#3d5a80", label="Calibrated", markersize=6)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("Reliability curve (test set)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(); sns.despine()
fig.savefig("assets/reliability_curve.png"); plt.close(fig)
print("saved assets/reliability_curve.png")
