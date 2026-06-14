"""Replicate leo.taps's idea: composite striker / grappler skill ratings.

Instead of feeding raw stats, fold them into z-scored skill scores per fighter
(striker, grappler, composite), then use the differences as features. This
sharpens the skill gap and should make the favourite-heavy calls more confident.
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

# Elo
rating = defaultdict(lambda: 1500.0); re_, be_ = [], []
for _, f in m.iterrows():
    rr, br = rating[f.R_fighter], rating[f.B_fighter]; re_.append(rr); be_.append(br)
    e = 1 / (1 + 10 ** ((br - rr) / 400)); s = 1.0 if f.Winner == "Red" else 0.0
    rating[f.R_fighter] = rr + 40 * (s - e); rating[f.B_fighter] = br + 40 * ((1 - s) - (1 - e))
m["R_elo"], m["B_elo"] = re_, be_

# z-score params for each skill component (pool R and B values)
COMP = {
    "sig_landed": ("avg_SIG_STR_landed", 1), "sig_pct": ("avg_SIG_STR_pct", 1), "ko_rate": (None, 1),
    "td_landed": ("avg_TD_landed", 1), "td_pct": ("avg_TD_pct", 1), "sub_att": ("avg_SUB_ATT", 1), "sub_rate": (None, 1),
}
def raw(side, key):
    if key == "ko_rate": return m[f"{side}_win_by_KO/TKO"] / m[f"{side}_wins"].clip(lower=1)
    if key == "sub_rate": return m[f"{side}_win_by_Submission"] / m[f"{side}_wins"].clip(lower=1)
    return m[f"{side}_{COMP[key][0]}"]

stats = {}
for key in COMP:
    pooled = pd.concat([raw("R", key), raw("B", key)])
    stats[key] = (pooled.mean(), pooled.std())

def z(side, key):
    mu, sd = stats[key]; return (raw(side, key) - mu) / sd

def skills(side):
    striker = z(side, "sig_landed") + z(side, "sig_pct") + z(side, "ko_rate")
    grappler = z(side, "td_landed") + z(side, "td_pct") + z(side, "sub_att") + z(side, "sub_rate")
    return striker, grappler

Rs, Rg = skills("R"); Bs, Bg = skills("B")
m["striker_dif"] = Rs - Bs; m["grappler_dif"] = Rg - Bg; m["composite_dif"] = (Rs + Rg) - (Bs + Bg)

# compute ALL difs ourselves as R - B (do NOT mix with the dataset's B-R prebuilt difs)
m["elo_dif"] = m.R_elo - m.B_elo
m["reach_dif"] = m.R_Reach_cms - m.B_Reach_cms
m["age_dif"] = m.R_age - m.B_age
m["win_streak_dif"] = m.R_current_win_streak - m.B_current_win_streak
m["win_pct_dif"] = m.R_wins / (m.R_wins + m.R_losses).clip(lower=1) - m.B_wins / (m.B_wins + m.B_losses).clip(lower=1)
m["y"] = (m.Winner == "Red").astype(int)

def bt(feats):
    d = m.dropna(subset=feats + ["y"]); c = int(len(d) * 0.8); a, b = d.iloc[:c], d.iloc[c:]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(a[feats], a.y)
    return accuracy_score(b.y, lr.predict(b[feats]))

BASE_F = ["elo_dif", "reach_dif", "age_dif", "win_streak_dif", "win_pct_dif"]
COMP_F = BASE_F + ["striker_dif", "grappler_dif", "composite_dif"]
print(f"base (5 feats):            {bt(BASE_F):.3f}")
print(f"+ composite skills (8):    {bt(COMP_F):.3f}")

# train final on composite features, predict the 4 fights
d = m.dropna(subset=COMP_F + ["y"])
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(d[COMP_F], d.y)

def latest_skills(fighter):
    rows = m[(m.R_fighter == fighter) | (m.B_fighter == fighter)]
    r = rows.iloc[-1]; p = "R" if r.R_fighter == fighter else "B"
    def zz(key):
        if key == "ko_rate": v = r[f"{p}_win_by_KO/TKO"] / max(r[f"{p}_wins"], 1)
        elif key == "sub_rate": v = r[f"{p}_win_by_Submission"] / max(r[f"{p}_wins"], 1)
        else: v = r[f"{p}_{COMP[key][0]}"]
        mu, sd = stats[key]; return (v - mu) / sd
    st = zz("sig_landed") + zz("sig_pct") + zz("ko_rate")
    gr = zz("td_landed") + zz("td_pct") + zz("sub_att") + zz("sub_rate")
    return dict(elo=rating[fighter], reach=r[f"{p}_Reach_cms"], age=r[f"{p}_age"],
                ws=r[f"{p}_current_win_streak"], wins=r[f"{p}_wins"], losses=r[f"{p}_losses"],
                striker=st, grappler=gr, comp=st + gr)

print(f"\n{'Fighter':18} {'Striker':>8} {'Grappler':>9} {'Composite':>10}")
for f in ["Ilia Topuria", "Justin Gaethje", "Ciryl Gane", "Alex Pereira", "Michael Chandler", "Mauricio Ruffy", "Aiemann Zahabi", "Sean O'Malley"]:
    s = latest_skills(f); print(f"{f:18} {s['striker']:>8.2f} {s['grappler']:>9.2f} {s['comp']:>10.2f}")

def _row(a, b):
    return pd.DataFrame([{"elo_dif": a["elo"] - b["elo"], "reach_dif": a["reach"] - b["reach"], "age_dif": a["age"] - b["age"],
        "win_streak_dif": a["ws"] - b["ws"], "win_pct_dif": a["wins"]/max(a["wins"]+a["losses"],1) - b["wins"]/max(b["wins"]+b["losses"],1),
        "striker_dif": a["striker"] - b["striker"], "grappler_dif": a["grappler"] - b["grappler"], "composite_dif": a["comp"] - b["comp"]}])[COMP_F]


def predict(f1, f2):
    # average over both corner assignments to cancel the Red-corner bias
    a, b = latest_skills(f1), latest_skills(f2)
    p_as_red = model.predict_proba(_row(a, b))[0][1]        # f1 in Red corner
    p_as_blue = 1 - model.predict_proba(_row(b, a))[0][1]    # f1 in Blue corner
    return (p_as_red + p_as_blue) / 2

print("\n=== predictions with composite skills ===")
for f1, f2, mk in [("Justin Gaethje","Ilia Topuria","Topuria 80%"),("Ciryl Gane","Alex Pereira","Pereira 51%"),
                   ("Michael Chandler","Mauricio Ruffy","Ruffy 81%"),("Aiemann Zahabi","Sean O'Malley","O'Malley 80%")]:
    p = predict(f1, f2); print(f"{f1} vs {f2}:  {f1} {p:.0%} / {f2} {1-p:.0%}   (market {mk})")
