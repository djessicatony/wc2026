"""SHAP waterfall plots — explain a single prediction (which features pushed it).

Uses XGBoost (the black box that needs SHAP; logreg you can just read via coef_).
Generates assets/shap_<match>.png for Brazil-Morocco and Netherlands-Japan.
"""

from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBClassifier

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"]).dropna(subset=["home_score"]).sort_values("date")

# Elo (leakage-safe), keep final ratings
ratings = defaultdict(lambda: 1500.0)
rows = []
for _, m in raw.iterrows():
    h, a = m.home_team, m.away_team; rh, ra = ratings[h], ratings[a]
    rows.append({"date": m.date, "home_team": h, "away_team": a, "home_elo": rh, "away_elo": ra})
    e = 1 / (1 + 10 ** ((ra - rh) / 400)); s = 1.0 if m.home_score > m.away_score else (0.5 if m.home_score == m.away_score else 0.0)
    ratings[h] = rh + 30 * (s - e); ratings[a] = ra + 30 * ((1 - s) - (1 - e))
elo = pd.DataFrame(rows)

df = pd.read_csv("data/training_set.csv", parse_dates=["date"]).merge(elo, on=["date", "home_team", "away_team"]).dropna()
RAWF = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf", "away_ga", "is_neutral", "home_elo", "away_elo"]
NICE = ["home win-rate", "home goals for", "home goals against", "away win-rate",
        "away goals for", "away goals against", "neutral ground", "home Elo", "away Elo"]

X = df[RAWF].copy(); X.columns = NICE
model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, eval_metric="logloss").fit(X, df["home_won"])
explainer = shap.TreeExplainer(model)


def current_form(team, window=10):
    m = raw[(raw.home_team == team) | (raw.away_team == team)].tail(window)
    gf = [r.home_score if r.home_team == team else r.away_score for _, r in m.iterrows()]
    ga = [r.away_score if r.home_team == team else r.home_score for _, r in m.iterrows()]
    won = [1 if f > g else 0 for f, g in zip(gf, ga)]
    n = len(m) or 1
    return sum(won) / n, sum(gf) / n, sum(ga) / n


def explain(home, away, fname):
    hf, af = current_form(home), current_form(away)
    row = pd.DataFrame([[hf[0], hf[1], hf[2], af[0], af[1], af[2], 1, ratings[home], ratings[away]]], columns=NICE)
    sv = explainer(row)
    plt.figure()
    shap.plots.waterfall(sv[0], max_display=9, show=False)
    plt.title(f"{home} vs {away} — why the model predicts what it does\n(positive → {home} win)", fontsize=11)
    plt.savefig(f"assets/{fname}.png", bbox_inches="tight", dpi=140)
    plt.close()
    print(f"saved assets/{fname}.png  (home={home}, P({home} win)={model.predict_proba(row)[0][1]:.1%})")


explain("Brazil", "Morocco", "shap_brazil_morocco")
explain("Netherlands", "Japan", "shap_netherlands_japan")
