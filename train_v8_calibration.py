"""v8: probability calibration — are the stated % honest?

Accuracy = how often right. Calibration = whether "30%" really means 30%.
We compare our probabilities to a market, so calibration matters.
Measured by Brier score (lower = better) + a reliability table.
Shows logreg is already well-calibrated; XGBoost is not, but can be fixed.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"])
raw = raw.dropna(subset=["home_score"]).sort_values("date")

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
df = pd.read_csv("data/training_set.csv", parse_dates=["date"]).merge(elo, on=keys).dropna().sort_values("date")
F = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf", "away_ga", "is_neutral", "home_elo", "away_elo"]
cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]
ytr, yte = tr["home_won"], te["home_won"]


def brier(model):
    p = model.fit(tr[F], ytr).predict_proba(te[F])[:, 1]
    return brier_score_loss(yte, p), p


logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, eval_metric="logloss")
# calibrated versions (fit calibrator via internal CV on train only -> no leakage)
logreg_cal = CalibratedClassifierCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), method="isotonic", cv=3)
xgb_cal = CalibratedClassifierCV(XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, eval_metric="logloss"), method="isotonic", cv=3)

print("=== BRIER SCORE (lower = better calibrated) ===")
b_lr, _ = brier(logreg);       print(f"logreg  raw:        {b_lr:.4f}")
b_lrc, _ = brier(logreg_cal);  print(f"logreg  calibrated: {b_lrc:.4f}")
b_xgb, p_xgb = brier(xgb);     print(f"XGBoost raw:        {b_xgb:.4f}")
b_xgbc, _ = brier(xgb_cal);    print(f"XGBoost calibrated: {b_xgbc:.4f}")

# reliability table for raw logreg: predicted prob vs actual frequency
print("\n=== RELIABILITY (logreg raw): does the % match reality? ===")
_, p_lr = brier(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
bins = np.linspace(0, 1, 6)
idx = np.digitize(p_lr, bins) - 1
print(f"{'predicted':>12} {'actual':>10} {'count':>7}")
for b in range(5):
    mask = idx == b
    if mask.sum():
        print(f"{p_lr[mask].mean():>11.0%} {yte.values[mask].mean():>10.0%} {mask.sum():>7}")
