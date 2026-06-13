"""v3: add an Elo rating feature (opponent strength) and re-measure.

Tests the hypothesis: the model underrated Brazil because it ignored
opponent quality. Elo encodes "who you beat" — beating a strong team
raises your rating a lot, beating a weak one barely. Leakage-safe: each
match uses the PRE-match Elo, then updates.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

K = 30  # how fast Elo reacts to a result

# ── compute pre-match Elo for every match, chronologically ──────────────
raw = pd.read_csv("data/international_results.csv", parse_dates=["date"])
raw = raw.dropna(subset=["home_score"]).sort_values("date")

ratings = defaultdict(lambda: 1500.0)
elo_rows = []
for _, r in raw.iterrows():
    h, a = r.home_team, r.away_team
    rh, ra = ratings[h], ratings[a]            # PRE-match ratings = the feature
    elo_rows.append({"date": r.date, "home_team": h, "away_team": a,
                     "home_elo": rh, "away_elo": ra})
    exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))   # expected score for home
    score_h = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    ratings[h] = rh + K * (score_h - exp_h)     # update after the match
    ratings[a] = ra + K * ((1 - score_h) - (1 - exp_h))

elo = pd.DataFrame(elo_rows)

# ── merge Elo onto the existing training set ────────────────────────────
df = pd.read_csv("data/training_set.csv", parse_dates=["date"])
df = df.merge(elo, on=["date", "home_team", "away_team"], how="left").dropna()

BASE = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf", "away_ga", "is_neutral"]
V3 = BASE + ["home_elo", "away_elo"]

# ── backtest both feature sets, same date split ─────────────────────────
df = df.sort_values("date")
cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]


def backtest(features):
    sc = StandardScaler()
    m = LogisticRegression(max_iter=1000).fit(sc.fit_transform(tr[features]), tr["home_won"])
    return accuracy_score(te["home_won"], m.predict(sc.transform(te[features])))


print("=== BACKTEST ===")
print(f"v1 (form only, 7 feat):       {backtest(BASE):.3f}")
print(f"v3 (form + Elo, 9 feat):      {backtest(V3):.3f}")

# ── re-predict Brazil vs Morocco with Elo ───────────────────────────────
sc = StandardScaler()
model = LogisticRegression(max_iter=1000).fit(sc.fit_transform(df[V3]), df["home_won"])


def current_form(team, window=10):
    m = raw[(raw.home_team == team) | (raw.away_team == team)].tail(window)
    gf = [r.home_score if r.home_team == team else r.away_score for _, r in m.iterrows()]
    ga = [r.away_score if r.home_team == team else r.home_score for _, r in m.iterrows()]
    won = [1 if f > g else 0 for f, g in zip(gf, ga)]
    n = len(m)
    return sum(won) / n, sum(gf) / n, sum(ga) / n


bra = current_form("Brazil"); mar = current_form("Morocco")
row = pd.DataFrame([{
    "home_win_rate": bra[0], "home_gf": bra[1], "home_ga": bra[2],
    "away_win_rate": mar[0], "away_gf": mar[1], "away_ga": mar[2],
    "is_neutral": 1,
    "home_elo": ratings["Brazil"], "away_elo": ratings["Morocco"],
}])[V3]
proba = model.predict_proba(sc.transform(row))[0]

print("\n=== PREDICTION: Brazil vs Morocco ===")
print(f"Brazil Elo: {ratings['Brazil']:.0f}   Morocco Elo: {ratings['Morocco']:.0f}")
print(f"P(Brazil wins) v3 = {proba[1]:.1%}   (v1 was 29.1%, market 59%)")
