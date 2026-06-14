"""Predict the FINISH METHOD (KO/TKO vs Submission vs Decision) — multiclass.

For 'how it ends' the signal is both fighters' tendencies combined (sums/avgs),
not who's better (differences). Heavy strikers -> KO; grapplers -> submission;
defensive / even -> decision.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

ORDER = {"Flyweight": 2, "Bantamweight": 3, "Featherweight": 4, "Lightweight": 5, "Welterweight": 6,
         "Middleweight": 7, "Light Heavyweight": 8, "Heavyweight": 9, "Women's Strawweight": 1,
         "Women's Flyweight": 2, "Women's Bantamweight": 3, "Women's Featherweight": 4, "Catch Weight": 5}

m = pd.read_csv("data/ufc_master.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
m = m[m.Winner.isin(["Red", "Blue"])].copy()

# Elo (for mismatch -> finish likelihood)
rating = defaultdict(lambda: 1500.0); re_, be_ = [], []
for _, f in m.iterrows():
    rr, br = rating[f.R_fighter], rating[f.B_fighter]; re_.append(rr); be_.append(br)
    e = 1 / (1 + 10 ** ((br - rr) / 400)); s = 1.0 if f.Winner == "Red" else 0.0
    rating[f.R_fighter] = rr + 40 * (s - e); rating[f.B_fighter] = br + 40 * ((1 - s) - (1 - e))
m["R_elo"], m["B_elo"] = re_, be_

# 3-class method label
MAP = {"KO/TKO": "KO/TKO", "SUB": "Submission", "U-DEC": "Decision", "S-DEC": "Decision", "M-DEC": "Decision"}
m["method3"] = m.finish.map(MAP)
m = m[m.method3.notna()].copy()


def feats(df):
    o = pd.DataFrame(index=df.index)
    ko = lambda p: df[f"{p}_win_by_KO/TKO"] / df[f"{p}_wins"].clip(lower=1)
    sub = lambda p: df[f"{p}_win_by_Submission"] / df[f"{p}_wins"].clip(lower=1)
    o["ko_tendency"] = (ko("R") + ko("B")) / 2          # both finish by strikes?
    o["sub_tendency"] = (sub("R") + sub("B")) / 2        # both submission-prone?
    o["striking_vol"] = df.R_avg_SIG_STR_landed + df.B_avg_SIG_STR_landed
    o["td_avg"] = (df.R_avg_TD_landed + df.B_avg_TD_landed) / 2
    o["sub_att_avg"] = (df.R_avg_SUB_ATT + df.B_avg_SUB_ATT) / 2
    o["weight"] = df.weight_class.map(ORDER)             # heavier -> more KOs
    o["elo_gap"] = (df.R_elo - df.B_elo).abs()           # mismatch -> finish
    o["age_avg"] = (df.R_age + df.B_age) / 2
    o["rounds"] = df.no_of_rounds
    return o


X = feats(m); y = m.method3
d = X.join(y).dropna(); cut = int(len(d) * 0.8)
tr, te = d.iloc[:cut], d.iloc[cut:]
F = list(X.columns)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[F], tr.method3)
acc = accuracy_score(te.method3, model.predict(te[F]))
base = accuracy_score(te.method3, [tr.method3.mode()[0]] * len(te))
print(f"method classes: {dict(y.value_counts())}")
print(f"\nbaseline (always '{tr.method3.mode()[0]}'): {base:.3f}")
print(f"our 3-class method model:           {acc:.3f}")

# ── predict method for the 4 fights ─────────────────────────────────────
def latest(fighter):
    r = m[(m.R_fighter == fighter) | (m.B_fighter == fighter)].iloc[-1]
    p = "R" if r.R_fighter == fighter else "B"; g = lambda c: r[f"{p}_{c}"]
    return dict(elo=rating[fighter], sig=g("avg_SIG_STR_landed"), td=g("avg_TD_landed"), sub=g("avg_SUB_ATT"),
                age=g("age"), ko=g("win_by_KO/TKO"), subw=g("win_by_Submission"), wins=g("wins"))


def predict_method(f1, f2, weight, rounds):
    a, b = latest(f1), latest(f2)
    row = pd.DataFrame([{
        "ko_tendency": (a["ko"] / max(a["wins"], 1) + b["ko"] / max(b["wins"], 1)) / 2,
        "sub_tendency": (a["subw"] / max(a["wins"], 1) + b["subw"] / max(b["wins"], 1)) / 2,
        "striking_vol": a["sig"] + b["sig"], "td_avg": (a["td"] + b["td"]) / 2,
        "sub_att_avg": (a["sub"] + b["sub"]) / 2, "weight": ORDER[weight],
        "elo_gap": abs(a["elo"] - b["elo"]), "age_avg": (a["age"] + b["age"]) / 2, "rounds": rounds,
    }])[F]
    p = model.predict_proba(row)[0]
    return dict(zip(model.classes_, p))


FIGHTS = [("Justin Gaethje", "Ilia Topuria", "Lightweight", 5),
          ("Ciryl Gane", "Alex Pereira", "Heavyweight", 5),
          ("Michael Chandler", "Mauricio Ruffy", "Lightweight", 3),
          ("Aiemann Zahabi", "Sean O'Malley", "Bantamweight", 5)]
print("\n=== predicted finish method ===")
for f1, f2, w, r in FIGHTS:
    p = predict_method(f1, f2, w, r)
    s = "  ".join(f"{k} {v:.0%}" for k, v in sorted(p.items(), key=lambda t: -t[1]))
    print(f"{f1} vs {f2}:  {s}")
