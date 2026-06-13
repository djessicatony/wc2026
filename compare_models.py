"""v2: compare logistic regression and XGBoost on the SAME data.

Honest experiment: does the complex model beat the simple one? We measure,
not guess. Comparison uses TimeSeriesSplit (time-aware k-fold) for a more
reliable number than a single split.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

df = pd.read_csv("data/training_set.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
F = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf", "away_ga", "is_neutral"]
X, y = df[F], df["home_won"]

tscv = TimeSeriesSplit(n_splits=5)
logreg_accs, xgb_accs = [], []

for tr, te in tscv.split(X):
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y.iloc[tr], y.iloc[te]

    # logistic regression — needs StandardScaler (scale matters for weights)
    sc = StandardScaler()
    lr = LogisticRegression().fit(sc.fit_transform(Xtr), ytr)
    logreg_accs.append(accuracy_score(yte, lr.predict(sc.transform(Xte))))

    # XGBoost — no scaling needed: trees split on thresholds, scale-invariant
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                        eval_metric="logloss")
    xgb.fit(Xtr, ytr)
    xgb_accs.append(accuracy_score(yte, xgb.predict(Xte)))

print("           logreg     XGBoost")
for i, (a, b) in enumerate(zip(logreg_accs, xgb_accs), 1):
    print(f"fold {i}:     {a:.3f}      {b:.3f}")
print(f"──────────────────────────────")
print(f"mean:      {np.mean(logreg_accs):.3f}      {np.mean(xgb_accs):.3f}")
