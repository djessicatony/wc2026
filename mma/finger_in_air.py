"""Finger-in-the-air: combine winner + method + round into one specific call.

Low confidence by design (10+ possible outcomes). For fun, not for betting.
P(outcome) ~ P(method+round) x P(winner is the finisher).
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ORDER = {"Flyweight": 2, "Bantamweight": 3, "Featherweight": 4, "Lightweight": 5, "Welterweight": 6,
         "Middleweight": 7, "Light Heavyweight": 8, "Heavyweight": 9}
m = pd.read_csv("data/ufc_master.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
m = m[m.Winner.isin(["Red", "Blue"])].copy()
rating = defaultdict(lambda: 1500.0); re_, be_ = [], []
for _, f in m.iterrows():
    rr, br = rating[f.R_fighter], rating[f.B_fighter]; re_.append(rr); be_.append(br)
    e = 1 / (1 + 10 ** ((br - rr) / 400)); s = 1.0 if f.Winner == "Red" else 0.0
    rating[f.R_fighter] = rr + 40 * (s - e); rating[f.B_fighter] = br + 40 * ((1 - s) - (1 - e))
m["R_elo"], m["B_elo"] = re_, be_

# ---- winner model (elo + age + win_pct) ----
m["elo_dif"] = m.R_elo - m.B_elo; m["age_dif"] = m.R_age - m.B_age
m["win_pct_dif"] = m.R_wins / (m.R_wins + m.R_losses).clip(lower=1) - m.B_wins / (m.B_wins + m.B_losses).clip(lower=1)
WF = ["elo_dif", "age_dif", "win_pct_dif"]
m["yw"] = (m.Winner == "Red").astype(int)
dw = m.dropna(subset=WF + ["yw"])
wmodel = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(dw[WF], dw.yw)

# ---- method+round model (5-class) ----
def lab(r):
    if r.finish in ("U-DEC", "S-DEC", "M-DEC"): return "Decision"
    meth = {"KO/TKO": "KO", "SUB": "Sub"}.get(r.finish)
    if meth and pd.notna(r.finish_round): return f"{meth} {'R1-2' if r.finish_round <= 2 else 'R3+'}"
    return None
m["lab"] = m.apply(lab, axis=1)
mm = m[m.lab.notna()].copy()
ko = lambda p, df: df[f"{p}_win_by_KO/TKO"] / df[f"{p}_wins"].clip(lower=1)
sub = lambda p, df: df[f"{p}_win_by_Submission"] / df[f"{p}_wins"].clip(lower=1)
def mfeats(df):
    return pd.DataFrame({"ko_t": (ko("R", df) + ko("B", df)) / 2, "sub_t": (sub("R", df) + sub("B", df)) / 2,
        "strv": df.R_avg_SIG_STR_landed + df.B_avg_SIG_STR_landed, "weight": df.weight_class.map(ORDER),
        "elo_gap": (df.R_elo - df.B_elo).abs(), "rounds": df.no_of_rounds}, index=df.index)
MF = ["ko_t", "sub_t", "strv", "weight", "elo_gap", "rounds"]
Xm = mfeats(mm).join(mm.lab).dropna()
mmodel = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xm[MF], Xm.lab)


def latest(f):
    r = m[(m.R_fighter == f) | (m.B_fighter == f)].iloc[-1]; p = "R" if r.R_fighter == f else "B"
    g = lambda c: r[f"{p}_{c}"]
    return dict(elo=rating[f], age=g("age"), wins=g("wins"), losses=g("losses"),
                ko=g("win_by_KO/TKO"), subw=g("win_by_Submission"), sig=g("avg_SIG_STR_landed"))


def call(f1, f2, weight, rounds):
    a, b = latest(f1), latest(f2)
    # winner (corner-averaged)
    def wrow(x, y): return pd.DataFrame([{"elo_dif": x["elo"]-y["elo"], "age_dif": x["age"]-y["age"],
        "win_pct_dif": x["wins"]/max(x["wins"]+x["losses"],1)-y["wins"]/max(y["wins"]+y["losses"],1)}])[WF]
    p1 = (wmodel.predict_proba(wrow(a, b))[0][1] + (1 - wmodel.predict_proba(wrow(b, a))[0][1])) / 2
    # method+round
    mr = pd.DataFrame([{"ko_t": (a["ko"]/max(a["wins"],1)+b["ko"]/max(b["wins"],1))/2,
        "sub_t": (a["subw"]/max(a["wins"],1)+b["subw"]/max(b["wins"],1))/2, "strv": a["sig"]+b["sig"],
        "weight": ORDER[weight], "elo_gap": abs(a["elo"]-b["elo"]), "rounds": rounds}])[MF]
    dist = dict(zip(mmodel.classes_, mmodel.predict_proba(mr)[0]))
    # aggregate round buckets -> method level (so KO isn't fragmented vs one Decision bucket)
    by_method = {"KO": dist.get("KO R1-2", 0) + dist.get("KO R3+", 0),
                 "Sub": dist.get("Sub R1-2", 0) + dist.get("Sub R3+", 0), "Decision": dist.get("Decision", 0)}
    print(f"\n{f1} vs {f2}")
    print(f"  winner: {f1} {p1:.0%} / {f2} {1-p1:.0%}")
    print(f"  by method:   " + "  ".join(f"{k} {v:.0%}" for k, v in sorted(by_method.items(), key=lambda t: -t[1])))
    print(f"  finish vs decision:  finish {1-by_method['Decision']:.0%}  /  decision {by_method['Decision']:.0%}")
    print(f"  by method+round: " + "  ".join(f"{k} {v:.0%}" for k, v in sorted(dist.items(), key=lambda t: -t[1])))
    # combine: attribute finish to the likely winner
    out = []
    for k, v in by_method.items():
        out.append((f"{f1} by {k}", v * p1)); out.append((f"{f2} by {k}", v * (1 - p1)))
    out.sort(key=lambda t: -t[1])
    print("  finger-in-the-air (winner x method, top 3):")
    for name, p in out[:3]: print(f"     {name}: {p:.0%}")


call("Ilia Topuria", "Justin Gaethje", "Lightweight", 5)
