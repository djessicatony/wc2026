"""Experiment: does a 'moved up/down in weight' feature help?

Expert nugget: Pereira is moving UP to heavyweight for the first time —
something our stats-only model is blind to. Test if a division-change
feature improves the backtest, and how it shifts Gane vs Pereira.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

ORDER = {"Women's Strawweight": 1, "Women's Flyweight": 2, "Women's Bantamweight": 3, "Women's Featherweight": 4,
         "Flyweight": 2, "Bantamweight": 3, "Featherweight": 4, "Lightweight": 5, "Welterweight": 6,
         "Middleweight": 7, "Light Heavyweight": 8, "Heavyweight": 9}

m = pd.read_csv("data/ufc_master.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
m = m[m.Winner.isin(["Red", "Blue"])].copy()
m["wc"] = m.weight_class.map(ORDER)

# chronological pass: Elo + last weight class per fighter -> division move
rating = defaultdict(lambda: 1500.0); last_wc = {}
re_, be_, rmv, bmv = [], [], [], []
for _, f in m.iterrows():
    rr, br = rating[f.R_fighter], rating[f.B_fighter]
    re_.append(rr); be_.append(br)
    rmv.append(f.wc - last_wc[f.R_fighter] if f.R_fighter in last_wc and pd.notna(f.wc) else 0)
    bmv.append(f.wc - last_wc[f.B_fighter] if f.B_fighter in last_wc and pd.notna(f.wc) else 0)
    e = 1 / (1 + 10 ** ((br - rr) / 400)); s = 1.0 if f.Winner == "Red" else 0.0
    rating[f.R_fighter] = rr + 40 * (s - e); rating[f.B_fighter] = br + 40 * ((1 - s) - (1 - e))
    if pd.notna(f.wc): last_wc[f.R_fighter] = f.wc; last_wc[f.B_fighter] = f.wc
m["R_elo"], m["B_elo"] = re_, be_
m["move_dif"] = np.array(rmv) - np.array(bmv)   # >0 = R moved up more than B


def base_feats(df):
    o = pd.DataFrame(index=df.index)
    o["elo_dif"] = df.R_elo - df.B_elo
    o["reach_dif"] = df.R_Reach_cms - df.B_Reach_cms
    o["age_dif"] = df.R_age - df.B_age
    o["sig_str_dif"] = df.R_avg_SIG_STR_landed - df.B_avg_SIG_STR_landed
    o["td_dif"] = df.R_avg_TD_landed - df.B_avg_TD_landed
    o["win_streak_dif"] = df.R_current_win_streak - df.B_current_win_streak
    o["win_pct_dif"] = df.R_wins / (df.R_wins + df.R_losses).clip(lower=1) - df.B_wins / (df.B_wins + df.B_losses).clip(lower=1)
    return o


X = base_feats(m); y = (m.Winner == "Red").astype(int)
d = X.join(m[["move_dif"]]).join(y.rename("y")).dropna()
cut = int(len(d) * 0.8); tr, te = d.iloc[:cut], d.iloc[cut:]
BASE = list(X.columns)


def acc(feats):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[feats], tr.y)
    return accuracy_score(te.y, lr.predict(te[feats]))


print(f"without move feature: {acc(BASE):.3f}")
print(f"with    move feature: {acc(BASE + ['move_dif']):.3f}")

# how many fights actually have a division move?
print(f"\nfights with a weight-class move: {(d.move_dif != 0).sum()} of {len(d)} ({(d.move_dif!=0).mean():.1%})")
w = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(d[BASE + ['move_dif']], d.y)
print(f"learned weight on move_dif: {w.named_steps['logisticregression'].coef_[0][-1]:+.3f}")
