"""MMA predictor — foundation + backtest vs the betting market.

Same playbook as football: fighter Elo + difference features -> predict the
winner -> backtest by date, compared against the market (favourite by odds).
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

m = pd.read_csv("data/ufc_master.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
m = m[m.Winner.isin(["Red", "Blue"])].copy()

# ── fighter Elo (leakage-safe): pre-fight rating is the feature ─────────
K = 40  # MMA has few fights per fighter -> faster adaptation than football
rating = defaultdict(lambda: 1500.0)
r_elo, b_elo = [], []
for _, f in m.iterrows():
    rr, br = rating[f.R_fighter], rating[f.B_fighter]
    r_elo.append(rr); b_elo.append(br)
    exp_r = 1 / (1 + 10 ** ((br - rr) / 400))
    sr = 1.0 if f.Winner == "Red" else 0.0
    rating[f.R_fighter] = rr + K * (sr - exp_r)
    rating[f.B_fighter] = br + K * ((1 - sr) - (1 - exp_r))
m["elo_dif"] = np.array(r_elo) - np.array(b_elo)

# ── features: our Elo + the dataset's prebuilt difference features ──────
FEATURES = ["elo_dif", "reach_dif", "age_dif", "sig_str_dif", "avg_td_dif",
            "avg_sub_att_dif", "win_streak_dif", "lose_streak_dif",
            "longest_win_streak_dif", "total_title_bout_dif", "height_dif"]
m["y"] = (m.Winner == "Red").astype(int)
d = m.dropna(subset=FEATURES + ["y"]).reset_index(drop=True)
print(f"fights for modelling: {len(d)}  ({d.date.min().date()} .. {d.date.max().date()})")

# ── backtest by date ────────────────────────────────────────────────────
cut = int(len(d) * 0.8)
tr, te = d.iloc[:cut], d.iloc[cut:]
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[FEATURES], tr.y)
xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, eval_metric="logloss").fit(tr[FEATURES], tr.y)
lr_acc = accuracy_score(te.y, lr.predict(te[FEATURES]))
xgb_acc = accuracy_score(te.y, xgb.predict(te[FEATURES]))

# market baseline: favourite (more negative American odds) wins
mk = te.dropna(subset=["R_odds", "B_odds"])
mk_pred = (mk.R_odds < mk.B_odds).astype(int)
mk_acc = accuracy_score(mk.y, mk_pred)

print(f"\n=== BACKTEST (test = {len(te)} recent fights) ===")
print(f"market (favourite wins): {mk_acc:.3f}   (on {len(mk)} fights with odds)")
print(f"logistic regression:     {lr_acc:.3f}")
print(f"XGBoost:                 {xgb_acc:.3f}")

print(f"\n=== what logreg learned (weights) ===")
w = lr.named_steps["logisticregression"].coef_[0]
for name, wi in sorted(zip(FEATURES, w), key=lambda t: -abs(t[1])):
    print(f"  {name:24} {wi:+.3f}")
