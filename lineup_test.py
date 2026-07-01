"""Manual check: would knowing the rested lineup have helped on these 2 matches?

We lower the resting team's Elo by a few deltas and watch the W/D/L shift, then
compare to what actually happened. 2 matches = anecdote, not proof — just a
directional smell test before building any pipeline.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEATURES = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf",
            "away_ga", "is_neutral", "elo_dif"]
WC_START = pd.Timestamp("2026-06-11")

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
    elo, on=["date", "home_team", "away_team"])
train = df[df.date < WC_START].dropna(subset=FEATURES + ["home_won"])
# 3-way target
res = raw.assign(result=np.where(raw.home_score > raw.away_score, "home",
                 np.where(raw.home_score < raw.away_score, "away", "draw")))
df = df.merge(res[["date", "home_team", "away_team", "result"]], on=["date", "home_team", "away_team"])
train = df[df.date < WC_START].dropna(subset=FEATURES + ["result"])
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(train[FEATURES], train["result"])
cls = list(model.classes_)


def predict(row, elo_shift_home=0.0):
    x = row[FEATURES].copy()
    x["elo_dif"] = x["elo_dif"] + elo_shift_home   # +shift = home stronger
    p = model.predict_proba(pd.DataFrame([x]))[0]
    return {c: p[i] for i, c in enumerate(cls)}


def show(home, away, rested_side, actual, deltas):
    row = df[(df.home_team == home) & (df.away_team == away) & (df.date >= WC_START)].iloc[0]
    print(f"\n=== {home} vs {away}   (отдыхала: {rested_side})   реально: {actual} ===")
    for d in deltas:
        # lower the resting team's strength: if home rested, home elo down -> elo_dif -d
        shift = -d if rested_side == home else +d   # rested=away -> home looks relatively stronger
        p = predict(row, elo_shift_home=shift)
        tag = "БАЗА" if d == 0 else f"-{d} {rested_side[:6]}"
        print(f"  {tag:14}  {home[:8]} {p['home']:.0%}  draw {p['draw']:.0%}  {away[:8]} {p['away']:.0%}")


show("Norway", "France", "Norway", "1-4 France (Франция разнесла)", [0, 80, 150])
show("Jordan", "Argentina", "Argentina", "1-3 Argentina (Аргентина уверенно)", [0, 60, 120])
