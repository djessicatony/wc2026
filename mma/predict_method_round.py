"""Predict method + round bucket — 5-class multiclass.

Classes: Decision / KO early (R1-2) / KO late (R3+) / Sub early / Sub late.
More granular than method alone -> harder -> lower accuracy (more classes,
data spread thinner). Demonstrates the 'finer label = less confident' rule.
"""

from collections import defaultdict
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
rating = defaultdict(lambda: 1500.0); re_, be_ = [], []
for _, f in m.iterrows():
    rr, br = rating[f.R_fighter], rating[f.B_fighter]; re_.append(rr); be_.append(br)
    e = 1 / (1 + 10 ** ((br - rr) / 400)); s = 1.0 if f.Winner == "Red" else 0.0
    rating[f.R_fighter] = rr + 40 * (s - e); rating[f.B_fighter] = br + 40 * ((1 - s) - (1 - e))
m["R_elo"], m["B_elo"] = re_, be_


def label(row):
    meth = {"KO/TKO": "KO", "SUB": "Sub"}.get(row.finish)
    if row.finish in ("U-DEC", "S-DEC", "M-DEC"):
        return "Decision"
    if meth and pd.notna(row.finish_round):
        return f"{meth} {'early' if row.finish_round <= 2 else 'late'}"
    return None


m["label"] = m.apply(label, axis=1)
m = m[m.label.notna()].copy()


def feats(df):
    o = pd.DataFrame(index=df.index)
    ko = lambda p: df[f"{p}_win_by_KO/TKO"] / df[f"{p}_wins"].clip(lower=1)
    sub = lambda p: df[f"{p}_win_by_Submission"] / df[f"{p}_wins"].clip(lower=1)
    o["ko_tendency"] = (ko("R") + ko("B")) / 2
    o["sub_tendency"] = (sub("R") + sub("B")) / 2
    o["striking_vol"] = df.R_avg_SIG_STR_landed + df.B_avg_SIG_STR_landed
    o["sub_att_avg"] = (df.R_avg_SUB_ATT + df.B_avg_SUB_ATT) / 2
    o["weight"] = df.weight_class.map(ORDER)
    o["elo_gap"] = (df.R_elo - df.B_elo).abs()
    o["age_avg"] = (df.R_age + df.B_age) / 2
    o["rounds"] = df.no_of_rounds
    return o


X = feats(m); d = X.join(m.label).dropna(); cut = int(len(d) * 0.8)
tr, te = d.iloc[:cut], d.iloc[cut:]; F = list(X.columns)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(tr[F], tr.label)
acc = accuracy_score(te.label, model.predict(te[F]))
base = accuracy_score(te.label, [tr.label.mode()[0]] * len(te))
print(f"classes: {dict(m.label.value_counts())}")
print(f"\nbaseline (always '{tr.label.mode()[0]}'): {base:.3f}")
print(f"our 5-class model:                  {acc:.3f}")


def latest(fighter):
    r = m[(m.R_fighter == fighter) | (m.B_fighter == fighter)].iloc[-1]
    p = "R" if r.R_fighter == fighter else "B"; g = lambda c: r[f"{p}_{c}"]
    return dict(elo=rating[fighter], sig=g("avg_SIG_STR_landed"), sub=g("avg_SUB_ATT"),
                age=g("age"), ko=g("win_by_KO/TKO"), subw=g("win_by_Submission"), wins=g("wins"))


def predict(f1, f2, weight, rounds):
    a, b = latest(f1), latest(f2)
    row = pd.DataFrame([{
        "ko_tendency": (a["ko"] / max(a["wins"], 1) + b["ko"] / max(b["wins"], 1)) / 2,
        "sub_tendency": (a["subw"] / max(a["wins"], 1) + b["subw"] / max(b["wins"], 1)) / 2,
        "striking_vol": a["sig"] + b["sig"], "sub_att_avg": (a["sub"] + b["sub"]) / 2,
        "weight": ORDER[weight], "elo_gap": abs(a["elo"] - b["elo"]), "age_avg": (a["age"] + b["age"]) / 2, "rounds": rounds,
    }])[F]
    return dict(zip(model.classes_, model.predict_proba(row)[0]))


FIGHTS = [("Justin Gaethje", "Ilia Topuria", "Lightweight", 5), ("Ciryl Gane", "Alex Pereira", "Heavyweight", 5),
          ("Michael Chandler", "Mauricio Ruffy", "Lightweight", 3), ("Aiemann Zahabi", "Sean O'Malley", "Bantamweight", 5)]
print("\n=== predicted method + round ===")
for f1, f2, w, r in FIGHTS:
    p = predict(f1, f2, w, r)
    s = "  ".join(f"{k} {v:.0%}" for k, v in sorted(p.items(), key=lambda t: -t[1])[:3])
    print(f"{f1} vs {f2}:  {s}")
