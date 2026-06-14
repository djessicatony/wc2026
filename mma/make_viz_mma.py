"""Generate MMA charts into mma/assets/ (same style as the football ones)."""

from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
TEAL, BLUE, ORANGE = "#2a9d8f", "#3d5a80", "#ee6c4d"

m = pd.read_csv("data/ufc_master.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
m = m[m.Winner.isin(["Red", "Blue"])].copy()

# K-independent skill composites
def rawv(side, key):
    if key == "ko": return m[f"{side}_win_by_KO/TKO"] / m[f"{side}_wins"].clip(lower=1)
    if key == "subr": return m[f"{side}_win_by_Submission"] / m[f"{side}_wins"].clip(lower=1)
    return m[f"{side}_{key}"]
KEYS = ["avg_SIG_STR_landed", "avg_SIG_STR_pct", "ko", "avg_TD_landed", "avg_TD_pct", "avg_SUB_ATT", "subr"]
ST = {k: (pd.concat([rawv("R", k), rawv("B", k)]).mean(), pd.concat([rawv("R", k), rawv("B", k)]).std()) for k in KEYS}
def z(side, k): return (rawv(side, k) - ST[k][0]) / ST[k][1]
sk = lambda s: z(s, "avg_SIG_STR_landed") + z(s, "avg_SIG_STR_pct") + z(s, "ko")
gr = lambda s: z(s, "avg_TD_landed") + z(s, "avg_TD_pct") + z(s, "avg_SUB_ATT") + z(s, "subr")
m["striker_dif"] = sk("R") - sk("B"); m["grappler_dif"] = gr("R") - gr("B")
m["composite_dif"] = (sk("R") + gr("R")) - (sk("B") + gr("B"))
m["reach_dif"] = m.R_Reach_cms - m.B_Reach_cms; m["age_dif"] = m.R_age - m.B_age
m["win_streak_dif"] = m.R_current_win_streak - m.B_current_win_streak
m["win_pct_dif"] = m.R_wins / (m.R_wins + m.R_losses).clip(lower=1) - m.B_wins / (m.B_wins + m.B_losses).clip(lower=1)
m["y"] = (m.Winner == "Red").astype(int)
FEATS = ["elo_dif", "reach_dif", "age_dif", "win_streak_dif", "win_pct_dif", "striker_dif", "grappler_dif", "composite_dif"]


def elo_pass(K):
    rating = defaultdict(lambda: 1500.0); col = []
    for _, f in m.iterrows():
        rr, br = rating[f.R_fighter], rating[f.B_fighter]; col.append(rr - br)
        e = 1 / (1 + 10 ** ((br - rr) / 400)); s = 1.0 if f.Winner == "Red" else 0.0
        rating[f.R_fighter] = rr + K * (s - e); rating[f.B_fighter] = br + K * ((1 - s) - (1 - e))
    return np.array(col), rating


def skills_of(fighter, rating):
    rows = m[(m.R_fighter == fighter) | (m.B_fighter == fighter)]; r = rows.iloc[-1]; p = "R" if r.R_fighter == fighter else "B"
    def zz(k):
        if k == "ko": v = r[f"{p}_win_by_KO/TKO"] / max(r[f"{p}_wins"], 1)
        elif k == "subr": v = r[f"{p}_win_by_Submission"] / max(r[f"{p}_wins"], 1)
        else: v = r[f"{p}_{k}"]
        return (v - ST[k][0]) / ST[k][1]
    s = zz("avg_SIG_STR_landed") + zz("avg_SIG_STR_pct") + zz("ko")
    g = zz("avg_TD_landed") + zz("avg_TD_pct") + zz("avg_SUB_ATT") + zz("subr")
    return dict(elo=rating[fighter], reach=r[f"{p}_Reach_cms"], age=r[f"{p}_age"], ws=r[f"{p}_current_win_streak"],
                wins=r[f"{p}_wins"], losses=r[f"{p}_losses"], st=s, gr=g, comp=s + g)


def featrow(a, b):
    return pd.DataFrame([{"elo_dif": a["elo"]-b["elo"], "reach_dif": a["reach"]-b["reach"], "age_dif": a["age"]-b["age"],
        "win_streak_dif": a["ws"]-b["ws"], "win_pct_dif": a["wins"]/max(a["wins"]+a["losses"],1)-b["wins"]/max(b["wins"]+b["losses"],1),
        "striker_dif": a["st"]-b["st"], "grappler_dif": a["gr"]-b["gr"], "composite_dif": a["comp"]-b["comp"]}])[FEATS]


def fit_predict(K, start_year, f1, f2):
    m["elo_dif"], rating = elo_pass(K)
    d = m[(m.date.dt.year >= start_year)].dropna(subset=FEATS + ["y"])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(d[FEATS], d.y)
    a, b = skills_of(f1, rating), skills_of(f2, rating)
    return (model.predict_proba(featrow(a, b))[0][1] + (1 - model.predict_proba(featrow(b, a))[0][1])) / 2, model, d


# ── 1. predictions vs market ────────────────────────────────────────────
FIGHTS = [("Ilia Topuria", "Justin Gaethje", 78, 80, 91), ("Alex Pereira", "Ciryl Gane", 44, 51, 62),
          ("Mauricio Ruffy", "Michael Chandler", 67, 81, None), ("Sean O'Malley", "Aiemann Zahabi", 61, 80, None)]
labels = [f"{a.split()[-1]}\nvs {b.split()[-1]}" for a, b, *_ in FIGHTS]
ours = [f[2] for f in FIGHTS]; mkt = [f[3] for f in FIGHTS]
x = np.arange(len(FIGHTS)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(x - w/2, ours, w, label="Our model", color=TEAL)
ax.bar(x + w/2, mkt, w, label="Polymarket", color=ORANGE)
for i, f in enumerate(FIGHTS):
    if f[4]: ax.scatter(i - w/2, f[4], color="#bc4749", zorder=5, s=60, marker="D", label="leo.taps" if i == 0 else "")
ax.axhline(50, ls="--", color="grey", lw=1, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("P(favourite wins) %"); ax.set_ylim(0, 100)
ax.set_title("MMA: our model is less confident in favourites than the market", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
for i, (o, mk) in enumerate(zip(ours, mkt)):
    ax.text(i - w/2, o + 1.5, f"{o}", ha="center", fontsize=9); ax.text(i + w/2, mk + 1.5, f"{mk}", ha="center", fontsize=9)
sns.despine(); fig.savefig("assets/mma_predictions.png"); plt.close(fig)

# ── 2. reliability curve ────────────────────────────────────────────────
m["elo_dif"], _ = elo_pass(40)
d = m.dropna(subset=FEATS + ["y"]); cut = int(len(d) * 0.8); tr, te = d.iloc[:cut], d.iloc[cut:]
rawm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[FEATS], tr.y)
calm = CalibratedClassifierCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), method="isotonic", cv=3).fit(tr[FEATS], tr.y)
yt_r, xp_r = calibration_curve(te.y, rawm.predict_proba(te[FEATS])[:, 1], n_bins=8)
yt_c, xp_c = calibration_curve(te.y, calm.predict_proba(te[FEATS])[:, 1], n_bins=8)
fig, ax = plt.subplots(figsize=(6.2, 6.2))
ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
ax.plot(xp_r, yt_r, "o-", color="#8d99ae", label="Raw")
ax.plot(xp_c, yt_c, "o-", color=BLUE, label="Calibrated")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("MMA reliability curve (test set)", fontsize=12, fontweight="bold"); ax.legend()
sns.despine(); fig.savefig("assets/mma_reliability.png"); plt.close(fig)

# ── 3. sensitivity heatmaps (Elo K x training-start-year) ───────────────
KS = [20, 30, 40, 50]; YEARS = [2010, 2014, 2018]
def heat(f1, f2, fname, title):
    g = np.zeros((len(KS), len(YEARS)))
    for i, K in enumerate(KS):
        for j, yr in enumerate(YEARS):
            g[i, j] = fit_predict(K, yr, f1, f2)[0] * 100
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sns.heatmap(g, annot=True, fmt=".0f", cmap="RdYlGn_r", center=50, vmin=20, vmax=80,
                xticklabels=[f"from {y}" for y in YEARS], yticklabels=[f"K={k}" for k in KS],
                cbar_kws={"label": f"P({f1.split()[-1]} wins) %"}, ax=ax, linewidths=1, linecolor="white")
    ax.set_title(f"{title}\n(range {g.min():.0f}–{g.max():.0f}%)", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"assets/{fname}.png", dpi=140); plt.close(fig)
    print(f"{f1} vs {f2}: {g.min():.0f}-{g.max():.0f}% (swing {g.max()-g.min():.0f})")

heat("Ilia Topuria", "Justin Gaethje", "mma_sensitivity_topuria", "Topuria vs Gaethje — robust favourite")
heat("Ciryl Gane", "Alex Pereira", "mma_sensitivity_gane", "Gane vs Pereira — coin-flip, less stable")
print("saved MMA charts to assets/")
