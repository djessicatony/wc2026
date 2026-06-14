"""MMA predictor v2 — add engineered features, measure vs v1 and the market."""

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

# ── Elo + layoff (both need a chronological pass) ───────────────────────
rating = defaultdict(lambda: 1500.0)
last_fight = {}
r_elo, b_elo, r_lay, b_lay = [], [], [], []
for _, f in m.iterrows():
    rr, br = rating[f.R_fighter], rating[f.B_fighter]
    r_elo.append(rr); b_elo.append(br)
    r_lay.append((f.date - last_fight[f.R_fighter]).days if f.R_fighter in last_fight else np.nan)
    b_lay.append((f.date - last_fight[f.B_fighter]).days if f.B_fighter in last_fight else np.nan)
    exp_r = 1 / (1 + 10 ** ((br - rr) / 400)); sr = 1.0 if f.Winner == "Red" else 0.0
    rating[f.R_fighter] = rr + 40 * (sr - exp_r); rating[f.B_fighter] = br + 40 * ((1 - sr) - (1 - exp_r))
    last_fight[f.R_fighter] = f.date; last_fight[f.B_fighter] = f.date
m["elo_dif"] = np.array(r_elo) - np.array(b_elo)
m["layoff_dif"] = np.array(r_lay) - np.array(b_lay)


# ── engineered difference features ──────────────────────────────────────
def finish_rate(p):
    return (m[f"{p}_win_by_KO/TKO"] + m[f"{p}_win_by_Submission"]) / m[f"{p}_wins"].clip(lower=1)
def win_pct(p):
    return m[f"{p}_wins"] / (m[f"{p}_wins"] + m[f"{p}_losses"]).clip(lower=1)

m["finish_rate_dif"] = finish_rate("R") - finish_rate("B")
m["win_pct_dif"] = win_pct("R") - win_pct("B")
m["experience_dif"] = (m.R_wins + m.R_losses) - (m.B_wins + m.B_losses)
m["stance_mismatch"] = (m.R_Stance != m.B_Stance).astype(int)

V1 = ["elo_dif", "reach_dif", "age_dif", "sig_str_dif", "avg_td_dif", "avg_sub_att_dif",
      "win_streak_dif", "lose_streak_dif", "longest_win_streak_dif", "total_title_bout_dif", "height_dif"]
NEW = ["finish_rate_dif", "win_pct_dif", "experience_dif", "layoff_dif", "stance_mismatch"]
m["y"] = (m.Winner == "Red").astype(int)

d = m.dropna(subset=V1 + ["y"]).reset_index(drop=True)
cut = int(len(d) * 0.8); tr, te = d.iloc[:cut], d.iloc[cut:]
mk = te.dropna(subset=["R_odds", "B_odds"])
mk_acc = accuracy_score(mk.y, (mk.R_odds < mk.B_odds).astype(int))


def bt(feats):
    t = d.dropna(subset=feats); c = int(len(t) * 0.8); a, b = t.iloc[:c], t.iloc[c:]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(a[feats].fillna(0), a.y)
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, eval_metric="logloss").fit(a[feats], a.y)
    return accuracy_score(b.y, lr.predict(b[feats].fillna(0))), accuracy_score(b.y, xg.predict(b[feats]))


print(f"market (favourite wins):     {mk_acc:.3f}")
lr1, xg1 = bt(V1); print(f"v1 (11 feats)  logreg {lr1:.3f}   XGB {xg1:.3f}")
lr2, xg2 = bt(V1 + NEW); print(f"v2 (16 feats)  logreg {lr2:.3f}   XGB {xg2:.3f}")
