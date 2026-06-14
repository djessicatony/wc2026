"""Predict any match with all models. Usage: python predict_match.py "Netherlands" "Japan"

Trains on form + Elo, runs logistic regression and XGBoost, both binary
(home win y/n) and 3-way (win/draw/loss). Assumes neutral ground (World Cup).
"""

import sys
from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

HOME = sys.argv[1] if len(sys.argv) > 1 else "Netherlands"
AWAY = sys.argv[2] if len(sys.argv) > 2 else "Japan"

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"])
raw = raw.dropna(subset=["home_score"]).sort_values("date")

# ── Elo (leakage-safe), keep final ratings for the prediction ───────────
ratings = defaultdict(lambda: 1500.0)
elo_rows = []
for _, r in raw.iterrows():
    h, a = r.home_team, r.away_team
    rh, ra = ratings[h], ratings[a]
    elo_rows.append({"date": r.date, "home_team": h, "away_team": a, "home_elo": rh, "away_elo": ra})
    exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
    sh = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    ratings[h] = rh + 30 * (sh - exp_h)
    ratings[a] = ra + 30 * ((1 - sh) - (1 - exp_h))
elo = pd.DataFrame(elo_rows)

keys = ["date", "home_team", "away_team"]
df = pd.read_csv("data/training_set.csv", parse_dates=["date"])
df = df.merge(raw[keys + ["home_score", "away_score"]], on=keys).merge(elo, on=keys).dropna()
df["result"] = (df.home_score > df.away_score).astype(int) * 2 + (df.home_score == df.away_score).astype(int)

F = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf", "away_ga", "is_neutral", "home_elo", "away_elo"]


def current_form(team, window=10):
    m = raw[(raw.home_team == team) | (raw.away_team == team)].tail(window)
    gf = [r.home_score if r.home_team == team else r.away_score for _, r in m.iterrows()]
    ga = [r.away_score if r.home_team == team else r.home_score for _, r in m.iterrows()]
    won = [1 if f > g else 0 for f, g in zip(gf, ga)]
    n = len(m) or 1
    return sum(won) / n, sum(gf) / n, sum(ga) / n


hf, af = current_form(HOME), current_form(AWAY)
row = pd.DataFrame([{
    "home_win_rate": hf[0], "home_gf": hf[1], "home_ga": hf[2],
    "away_win_rate": af[0], "away_gf": af[1], "away_ga": af[2],
    "is_neutral": 1, "home_elo": ratings[HOME], "away_elo": ratings[AWAY],
}])[F]

print(f"=== {HOME} vs {AWAY} ===")
print(f"Elo: {HOME} {ratings[HOME]:.0f}   {AWAY} {ratings[AWAY]:.0f}")
print(f"form {HOME}: wr={hf[0]:.2f} gf={hf[1]:.2f} ga={hf[2]:.2f}")
print(f"form {AWAY}: wr={af[0]:.2f} gf={af[1]:.2f} ga={af[2]:.2f}\n")

# binary: P(home win)
lr_b = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(df[F], df["home_won"])
xgb_b = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, eval_metric="logloss").fit(df[F], df["home_won"])
print(f"BINARY  P({HOME} wins):  logreg {lr_b.predict_proba(row)[0][1]:.1%}   XGBoost {xgb_b.predict_proba(row)[0][1]:.1%}")

# 3-way: [away win, draw, home win]
lr_3 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(df[F], df["result"])
xgb_3 = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, eval_metric="logloss").fit(df[F], df["result"])
pl, px = lr_3.predict_proba(row)[0], xgb_3.predict_proba(row)[0]
print(f"\n3-WAY            logreg    XGBoost")
print(f"  {HOME+' win':14} {pl[2]:>6.1%}    {px[2]:>6.1%}")
print(f"  {'Draw':14} {pl[1]:>6.1%}    {px[1]:>6.1%}")
print(f"  {AWAY+' win':14} {pl[0]:>6.1%}    {px[0]:>6.1%}")
