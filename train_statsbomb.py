"""v2: train on rich StatsBomb features + compare to v1.

Same structure as v1: form (rolling) -> one row per match -> model.
But features are rich (xG, possession, accuracy) and matches are few (~300).
Question: do rich features beat our v1 65% despite the small sample?
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

WINDOW = 5  # few tournament matches -> smaller form window than v1
RICH = ["xg_for", "xg_against", "goals_for", "goals_against",
        "shot_accuracy", "possession", "won"]

long = pd.read_csv("data/statsbomb_team_matches.csv", parse_dates=["date"])
long = long.sort_values(["team", "date"]).reset_index(drop=True)

# form over the last WINDOW matches (shift(1) = no leakage, same as v1)
grp = long.groupby("team")
for col in RICH:
    long[f"form_{col}"] = grp[col].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())

FORM = [f"form_{c}" for c in RICH]

# one row per match: home form + away form (two sides share a match_id)
rows = []
for mid, g in long.groupby("match_id"):
    if len(g) != 2:
        continue
    a, b = g.iloc[0], g.iloc[1]
    row = {"date": a["date"]}
    for c in FORM:
        row[f"home_{c}"] = a[c]
        row[f"away_{c}"] = b[c]
    row["home_won"] = a["won"]
    rows.append(row)

data = pd.DataFrame(rows).dropna().sort_values("date").reset_index(drop=True)
FEATURES = [f"home_{c}" for c in FORM] + [f"away_{c}" for c in FORM]
X, y = data[FEATURES], data["home_won"]
print(f"training matches: {len(data)}, features: {len(FEATURES)}")

# compare logreg vs XGBoost via time-series k-fold
tscv = TimeSeriesSplit(n_splits=5)
lr_accs, xgb_accs = [], []
for tr, te in tscv.split(X):
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=1000).fit(sc.fit_transform(X.iloc[tr]), y.iloc[tr])
    lr_accs.append(accuracy_score(y.iloc[te], lr.predict(sc.transform(X.iloc[te]))))
    xgb = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, eval_metric="logloss")
    xgb.fit(X.iloc[tr], y.iloc[tr])
    xgb_accs.append(accuracy_score(y.iloc[te], xgb.predict(X.iloc[te])))

print(f"\n=== v2 (rich StatsBomb features, {len(data)} matches) ===")
print(f"logreg:  {np.mean(lr_accs):.3f}")
print(f"XGBoost: {np.mean(xgb_accs):.3f}")
print(f"\n=== v1 (simple features, 25080 matches) for comparison ===")
print(f"logreg:  0.645")
print(f"XGBoost: 0.641")
