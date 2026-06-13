"""v7: add match-importance (competition tier) as a feature.

Tests whether importance helps as (a) a flat feature for logreg, and
(b) for XGBoost, which can use it as an interaction with Elo
("trust Elo more in serious matches").
"""

from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"])
raw = raw.dropna(subset=["home_score"]).sort_values("date")


def tier(t: str) -> int:
    t = str(t).lower()
    if "world cup" in t and "qualif" not in t:
        return 5
    if any(x in t for x in ["euro", "copa", "african cup", "asian cup", "gold cup"]) and "qualif" not in t:
        return 4
    if "qualif" in t:
        return 3
    if "nations league" in t:
        return 3
    if "friendly" in t:
        return 1
    return 2


raw["importance"] = raw["tournament"].map(tier)

# simple Elo (leakage-safe)
ratings = defaultdict(lambda: 1500.0)
rows = []
for _, r in raw.iterrows():
    h, a = r.home_team, r.away_team
    rh, ra = ratings[h], ratings[a]
    rows.append({"date": r.date, "home_team": h, "away_team": a, "home_elo": rh, "away_elo": ra})
    exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
    sh = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    ratings[h] = rh + 30 * (sh - exp_h)
    ratings[a] = ra + 30 * ((1 - sh) - (1 - exp_h))
elo = pd.DataFrame(rows)

keys = ["date", "home_team", "away_team"]
df = pd.read_csv("data/training_set.csv", parse_dates=["date"])
df = df.merge(raw[keys + ["importance"]], on=keys).merge(elo, on=keys).dropna().sort_values("date")

BASE = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf", "away_ga", "is_neutral", "home_elo", "away_elo"]
cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]


def acc(model, features):
    if isinstance(model, LogisticRegression):
        sc = StandardScaler()
        model.fit(sc.fit_transform(tr[features]), tr["home_won"])
        return accuracy_score(te["home_won"], model.predict(sc.transform(te[features])))
    model.fit(tr[features], tr["home_won"])
    return accuracy_score(te["home_won"], model.predict(te[features]))


def xgb():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, eval_metric="logloss")


print("=== does match-importance help? (binary) ===")
print(f"logreg,  form+Elo:             {acc(LogisticRegression(max_iter=1000), BASE):.3f}")
print(f"logreg,  form+Elo+importance:  {acc(LogisticRegression(max_iter=1000), BASE + ['importance']):.3f}")
print(f"XGBoost, form+Elo:             {acc(xgb(), BASE):.3f}")
print(f"XGBoost, form+Elo+importance:  {acc(xgb(), BASE + ['importance']):.3f}")
