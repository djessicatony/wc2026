"""Group-stage backtest chart for the README: the draw problem.

Left  — model P(draw) for matches that drew vs matches that didn't (overlapping
         distributions = the model can't tell them apart).
Right — actual vs model-expected draw rate, and winner accuracy vs the trivial
         Elo baseline.
Runs the same leakage-safe backtest as wc_accuracy.py.
"""

from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
TEAL, BLUE, ORANGE = "#2a9d8f", "#3d5a80", "#ee6c4d"

WC_START = pd.Timestamp("2026-06-11")
FEATURES = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf",
            "away_ga", "is_neutral", "elo_dif"]

raw = (pd.read_csv("data/international_results.csv", parse_dates=["date"])
       .dropna(subset=["home_score"]).sort_values("date"))
rating = defaultdict(lambda: 1500.0); rows = []
for _, r in raw.iterrows():
    h, a = r.home_team, r.away_team; rh, ra = rating[h], rating[a]
    rows.append({"date": r.date, "home_team": h, "away_team": a, "elo_dif": rh - ra})
    e = 1 / (1 + 10 ** ((ra - rh) / 400))
    s = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    rating[h] = rh + 30 * (s - e); rating[a] = ra + 30 * ((1 - s) - (1 - e))
elo = pd.DataFrame(rows)

df = pd.read_csv("data/training_set.csv", parse_dates=["date"]).merge(
    elo, on=["date", "home_team", "away_team"]).merge(
    raw[["date", "home_team", "away_team", "home_score", "away_score"]],
    on=["date", "home_team", "away_team"])
df["result"] = np.where(df.home_score > df.away_score, "home",
                np.where(df.home_score < df.away_score, "away", "draw"))
train = df[df.date < WC_START].dropna(subset=FEATURES + ["result"])
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
    train[FEATURES], train["result"])
classes = list(model.classes_)

wc = df[df.date >= WC_START].dropna(subset=FEATURES + ["result"]).copy()
probs = model.predict_proba(wc[FEATURES])
wc["p_draw"] = probs[:, classes.index("draw")]
wc["pick"] = [classes[i] for i in probs.argmax(1)]
n = len(wc)
acc = (wc.pick == wc.result).mean() * 100
base = (np.where(wc.elo_dif >= 0, "home", "away") == wc.result).mean() * 100
actual_draw = (wc.result == "draw").mean() * 100
exp_draw = wc.p_draw.mean() * 100

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

# left: P(draw) distributions
bins = np.linspace(0, 0.45, 13)
axL.hist(wc.loc[wc.result == "draw", "p_draw"], bins=bins, color=ORANGE, alpha=0.7,
         label=f"actually drew (n={int((wc.result=='draw').sum())})", edgecolor="white")
axL.hist(wc.loc[wc.result != "draw", "p_draw"], bins=bins, color=BLUE, alpha=0.55,
         label=f"did not draw (n={int((wc.result!='draw').sum())})", edgecolor="white")
axL.axvline(1/3, ls="--", color="#888", lw=1)
axL.text(1/3 + .005, axL.get_ylim()[1]*0.9, "33% (argmax\nthreshold)", fontsize=9, color="#555")
axL.set_xlabel("model P(draw)"); axL.set_ylabel("matches")
axL.set_title("The model can't spot a draw\n(both groups sit at the same low P)", fontsize=13)
axL.legend(fontsize=10)

# right: summary bars
labels = ["winner\naccuracy", "Elo-baseline\naccuracy", "actual\ndraw rate", "model expected\ndraw rate"]
vals = [acc, base, actual_draw, exp_draw]
colors = [TEAL, "#b8c4d0", ORANGE, "#f4a98f"]
bars = axR.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
for b, v in zip(bars, vals):
    axR.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", fontsize=11, color="#333")
axR.set_ylim(0, 75); axR.set_ylabel("%")
axR.set_title(f"All {n} group-stage matches, out-of-sample\naccuracy ties the baseline; draws under-counted", fontsize=13)

fig.tight_layout()
fig.savefig("assets/wc_draw_analysis.png")
print("saved assets/wc_draw_analysis.png")
