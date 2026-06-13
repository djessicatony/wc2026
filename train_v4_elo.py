"""v4: smarter Elo (match importance + goal difference), eloratings.net style.

Compares three feature sets in one run:
  v1 = form only
  v3 = form + simple Elo (flat K=30)
  v4 = form + weighted Elo (importance + goal-diff weights)
All Elo variants use data we already have (tournament name, scores).
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"])
raw = raw.dropna(subset=["home_score"]).sort_values("date")


def importance_k(tournament: str) -> int:
    t = str(tournament).lower()
    if "world cup" in t and "qualif" not in t:
        return 60
    if any(x in t for x in ["euro", "copa", "african", "gold cup", "asian cup", "confederations"]):
        return 50 if "qualif" not in t else 40
    if "qualif" in t or "nations league" in t:
        return 40
    if "friendly" in t:
        return 20
    return 30


def goal_diff_mult(diff: int) -> float:
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return (11 + diff) / 8  # eloratings formula for 3+ goal margins


def run_elo(weighted: bool):
    ratings = defaultdict(lambda: 1500.0)
    rows = []
    for _, r in raw.iterrows():
        h, a = r.home_team, r.away_team
        rh, ra = ratings[h], ratings[a]
        rows.append({"date": r.date, "home_team": h, "away_team": a,
                     "home_elo": rh, "away_elo": ra})
        exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
        diff = abs(r.home_score - r.away_score)
        score_h = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        k = importance_k(r.tournament) * goal_diff_mult(diff) if weighted else 30
        ratings[h] = rh + k * (score_h - exp_h)
        ratings[a] = ra + k * ((1 - score_h) - (1 - exp_h))
    return pd.DataFrame(rows), ratings


elo_simple, rat_simple = run_elo(weighted=False)
elo_weighted, rat_weighted = run_elo(weighted=True)

df = pd.read_csv("data/training_set.csv", parse_dates=["date"])
keys = ["date", "home_team", "away_team"]
df = (df.merge(elo_simple.rename(columns={"home_elo": "home_elo_s", "away_elo": "away_elo_s"}), on=keys)
        .merge(elo_weighted.rename(columns={"home_elo": "home_elo_w", "away_elo": "away_elo_w"}), on=keys)
        .dropna().sort_values("date"))

BASE = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf", "away_ga", "is_neutral"]
cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]


def backtest(features):
    sc = StandardScaler()
    m = LogisticRegression(max_iter=1000).fit(sc.fit_transform(tr[features]), tr["home_won"])
    return accuracy_score(te["home_won"], m.predict(sc.transform(te[features])))


print("=== BACKTEST ===")
print(f"v1 (form only):              {backtest(BASE):.3f}")
print(f"v3 (form + simple Elo):      {backtest(BASE + ['home_elo_s', 'away_elo_s']):.3f}")
print(f"v4 (form + weighted Elo):    {backtest(BASE + ['home_elo_w', 'away_elo_w']):.3f}")

print("\n=== Elo: Brazil vs Morocco ===")
print(f"simple:   Brazil {rat_simple['Brazil']:.0f}  Morocco {rat_simple['Morocco']:.0f}")
print(f"weighted: Brazil {rat_weighted['Brazil']:.0f}  Morocco {rat_weighted['Morocco']:.0f}")
