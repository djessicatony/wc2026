"""Add a generalizable stance-interaction feature (southpaw edge) on top of
the composite-skills model. Measure if it helps, and how it moves the fights.

southpaw_edge (R - B convention): +1 if R is southpaw vs orthodox B,
-1 if R orthodox vs southpaw B, 0 otherwise. Captures the known small
southpaw striking advantage in a generalizable, antisymmetric way.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

m = pd.read_csv("data/ufc_master.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
m = m[m.Winner.isin(["Red", "Blue"])].copy()
rating = defaultdict(lambda: 1500.0); re_, be_ = [], []
for _, f in m.iterrows():
    rr, br = rating[f.R_fighter], rating[f.B_fighter]; re_.append(rr); be_.append(br)
    e = 1 / (1 + 10 ** ((br - rr) / 400)); s = 1.0 if f.Winner == "Red" else 0.0
    rating[f.R_fighter] = rr + 40 * (s - e); rating[f.B_fighter] = br + 40 * ((1 - s) - (1 - e))
m["R_elo"], m["B_elo"] = re_, be_

# z-scored skill composites (same as before)
def raw(side, key):
    if key == "ko": return m[f"{side}_win_by_KO/TKO"] / m[f"{side}_wins"].clip(lower=1)
    if key == "subr": return m[f"{side}_win_by_Submission"] / m[f"{side}_wins"].clip(lower=1)
    return m[f"{side}_{key}"]
KEYS = {"avg_SIG_STR_landed": 0, "avg_SIG_STR_pct": 0, "ko": 0, "avg_TD_landed": 0, "avg_TD_pct": 0, "avg_SUB_ATT": 0, "subr": 0}
st = {k: (pd.concat([raw("R", k), raw("B", k)]).mean(), pd.concat([raw("R", k), raw("B", k)]).std()) for k in KEYS}
def z(side, k): return (raw(side, k) - st[k][0]) / st[k][1]
def striker(s): return z(s, "avg_SIG_STR_landed") + z(s, "avg_SIG_STR_pct") + z(s, "ko")
def grappler(s): return z(s, "avg_TD_landed") + z(s, "avg_TD_pct") + z(s, "avg_SUB_ATT") + z(s, "subr")
m["striker_dif"] = striker("R") - striker("B")
m["grappler_dif"] = grappler("R") - grappler("B")
m["composite_dif"] = (striker("R") + grappler("R")) - (striker("B") + grappler("B"))

# stance edge (directional southpaw advantage)
def edge(rs, bs):
    if rs == "Southpaw" and bs == "Orthodox": return 1
    if rs == "Orthodox" and bs == "Southpaw": return -1
    return 0
m["southpaw_edge"] = [edge(r, b) for r, b in zip(m.R_Stance, m.B_Stance)]

m["elo_dif"] = m.R_elo - m.B_elo
m["reach_dif"] = m.R_Reach_cms - m.B_Reach_cms
m["age_dif"] = m.R_age - m.B_age
m["win_streak_dif"] = m.R_current_win_streak - m.B_current_win_streak
m["win_pct_dif"] = m.R_wins / (m.R_wins + m.R_losses).clip(lower=1) - m.B_wins / (m.B_wins + m.B_losses).clip(lower=1)
m["y"] = (m.Winner == "Red").astype(int)

COMP = ["elo_dif", "reach_dif", "age_dif", "win_streak_dif", "win_pct_dif", "striker_dif", "grappler_dif", "composite_dif"]
def bt(feats):
    d = m.dropna(subset=feats + ["y"]); c = int(len(d) * 0.8); a, b = d.iloc[:c], d.iloc[c:]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(a[feats], a.y)
    return accuracy_score(b.y, lr.predict(b[feats]))
print(f"composite (8 feats):        {bt(COMP):.3f}")
print(f"+ southpaw_edge (9 feats):   {bt(COMP + ['southpaw_edge']):.3f}")

F = COMP + ["southpaw_edge"]
d = m.dropna(subset=F + ["y"])
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(d[F], d.y)
print(f"southpaw_edge weight: {model.named_steps['logisticregression'].coef_[0][-1]:+.3f}")

def latest(fighter):
    rows = m[(m.R_fighter == fighter) | (m.B_fighter == fighter)]; r = rows.iloc[-1]; p = "R" if r.R_fighter == fighter else "B"
    def zz(k):
        if k == "ko": v = r[f"{p}_win_by_KO/TKO"] / max(r[f"{p}_wins"], 1)
        elif k == "subr": v = r[f"{p}_win_by_Submission"] / max(r[f"{p}_wins"], 1)
        else: v = r[f"{p}_{k}"]
        return (v - st[k][0]) / st[k][1]
    sk = zz("avg_SIG_STR_landed") + zz("avg_SIG_STR_pct") + zz("ko")
    gr = zz("avg_TD_landed") + zz("avg_TD_pct") + zz("avg_SUB_ATT") + zz("subr")
    return dict(elo=rating[fighter], reach=r[f"{p}_Reach_cms"], age=r[f"{p}_age"], ws=r[f"{p}_current_win_streak"],
                wins=r[f"{p}_wins"], losses=r[f"{p}_losses"], stance=r[f"{p}_Stance"], st=sk, gr=gr, comp=sk + gr)

def row(a, b):
    return pd.DataFrame([{"elo_dif": a["elo"]-b["elo"], "reach_dif": a["reach"]-b["reach"], "age_dif": a["age"]-b["age"],
        "win_streak_dif": a["ws"]-b["ws"], "win_pct_dif": a["wins"]/max(a["wins"]+a["losses"],1)-b["wins"]/max(b["wins"]+b["losses"],1),
        "striker_dif": a["st"]-b["st"], "grappler_dif": a["gr"]-b["gr"], "composite_dif": a["comp"]-b["comp"],
        "southpaw_edge": edge(a["stance"], b["stance"])}])[F]
def predict(f1, f2):
    a, b = latest(f1), latest(f2)
    return (model.predict_proba(row(a, b))[0][1] + (1 - model.predict_proba(row(b, a))[0][1])) / 2

print("\n=== with stance feature ===")
for f1, f2, mk in [("Justin Gaethje","Ilia Topuria","Topuria 80%"),("Ciryl Gane","Alex Pereira","Pereira 51%"),
                   ("Michael Chandler","Mauricio Ruffy","Ruffy 81%"),("Aiemann Zahabi","Sean O'Malley","O'Malley 80%")]:
    a, b = latest(f1), latest(f2); p = predict(f1, f2)
    print(f"{f1}({a['stance']}) vs {f2}({b['stance']}):  {f1} {p:.0%} / {f2} {1-p:.0%}   (market {mk})")
