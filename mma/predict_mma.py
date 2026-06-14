"""Train the MMA model and predict the 4 June 15 fights.

All difference features are computed ourselves as (R - B) for a single
consistent convention, so training and prediction match exactly.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

m = pd.read_csv("data/ufc_master.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
m = m[m.Winner.isin(["Red", "Blue"])].copy()

# chronological pass: Elo + last-fight date
rating = defaultdict(lambda: 1500.0); last = {}
re_, be_, rl_, bl_ = [], [], [], []
for _, f in m.iterrows():
    rr, br = rating[f.R_fighter], rating[f.B_fighter]
    re_.append(rr); be_.append(br)
    rl_.append((f.date - last[f.R_fighter]).days if f.R_fighter in last else np.nan)
    bl_.append((f.date - last[f.B_fighter]).days if f.B_fighter in last else np.nan)
    e = 1 / (1 + 10 ** ((br - rr) / 400)); s = 1.0 if f.Winner == "Red" else 0.0
    rating[f.R_fighter] = rr + 40 * (s - e); rating[f.B_fighter] = br + 40 * ((1 - s) - (1 - e))
    last[f.R_fighter] = f.date; last[f.B_fighter] = f.date
m["R_elo"], m["B_elo"], m["R_lay"], m["B_lay"] = re_, be_, rl_, bl_


def difs(df):
    """all features as R - B (one consistent convention)"""
    out = pd.DataFrame()
    out["elo_dif"] = df.R_elo - df.B_elo
    out["reach_dif"] = df.R_Reach_cms - df.B_Reach_cms
    out["age_dif"] = df.R_age - df.B_age
    out["height_dif"] = df.R_Height_cms - df.B_Height_cms
    out["sig_str_dif"] = df.R_avg_SIG_STR_landed - df.B_avg_SIG_STR_landed
    out["td_dif"] = df.R_avg_TD_landed - df.B_avg_TD_landed
    out["sub_att_dif"] = df.R_avg_SUB_ATT - df.B_avg_SUB_ATT
    out["win_streak_dif"] = df.R_current_win_streak - df.B_current_win_streak
    out["lose_streak_dif"] = df.R_current_lose_streak - df.B_current_lose_streak
    out["title_dif"] = df.R_total_title_bouts - df.B_total_title_bouts
    out["finish_rate_dif"] = ((df["R_win_by_KO/TKO"] + df.R_win_by_Submission) / df.R_wins.clip(lower=1)
                              - (df["B_win_by_KO/TKO"] + df.B_win_by_Submission) / df.B_wins.clip(lower=1))
    out["win_pct_dif"] = (df.R_wins / (df.R_wins + df.R_losses).clip(lower=1)
                          - df.B_wins / (df.B_wins + df.B_losses).clip(lower=1))
    out["exp_dif"] = (df.R_wins + df.R_losses) - (df.B_wins + df.B_losses)
    out["layoff_dif"] = df.R_lay - df.B_lay
    out["stance_mismatch"] = (df.R_Stance != df.B_Stance).astype(int)
    return out


X = difs(m); y = (m.Winner == "Red").astype(int)
ok = X.dropna().index
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X.loc[ok], y.loc[ok])

# ── per-fighter latest self-stats ───────────────────────────────────────
def latest(fighter):
    rows = m[(m.R_fighter == fighter) | (m.B_fighter == fighter)]
    r = rows.iloc[-1]; p = "R" if r.R_fighter == fighter else "B"
    g = lambda c: r[f"{p}_{c}"]
    return dict(elo=rating[fighter], reach=g("Reach_cms"), age=g("age"), height=g("Height_cms"),
                sig=g("avg_SIG_STR_landed"), td=g("avg_TD_landed"), sub=g("avg_SUB_ATT"),
                ws=g("current_win_streak"), ls=g("current_lose_streak"), title=g("total_title_bouts"),
                ko=g("win_by_KO/TKO"), subw=g("win_by_Submission"), wins=g("wins"), losses=g("losses"),
                stance=g("Stance"), last=last[fighter])


FIGHT_DATE = pd.Timestamp("2026-06-15")


def predict(f1, f2):
    a, b = latest(f1), latest(f2)
    fr = lambda x: (x["ko"] + x["subw"]) / max(x["wins"], 1)
    row = pd.DataFrame([{
        "elo_dif": a["elo"] - b["elo"], "reach_dif": a["reach"] - b["reach"], "age_dif": a["age"] - b["age"],
        "height_dif": a["height"] - b["height"], "sig_str_dif": a["sig"] - b["sig"], "td_dif": a["td"] - b["td"],
        "sub_att_dif": a["sub"] - b["sub"], "win_streak_dif": a["ws"] - b["ws"], "lose_streak_dif": a["ls"] - b["ls"],
        "title_dif": a["title"] - b["title"], "finish_rate_dif": fr(a) - fr(b),
        "win_pct_dif": a["wins"] / max(a["wins"] + a["losses"], 1) - b["wins"] / max(b["wins"] + b["losses"], 1),
        "exp_dif": (a["wins"] + a["losses"]) - (b["wins"] + b["losses"]),
        "layoff_dif": (FIGHT_DATE - a["last"]).days - (FIGHT_DATE - b["last"]).days,
        "stance_mismatch": int(a["stance"] != b["stance"]),
    }])[X.columns]
    return model.predict_proba(row)[0][1]


FIGHTS = [("Justin Gaethje", "Ilia Topuria"), ("Ciryl Gane", "Alex Pereira"),
          ("Michael Chandler", "Mauricio Ruffy"), ("Aiemann Zahabi", "Sean O'Malley")]
MARKET = {"Ilia Topuria": 80, "Alex Pereira": 51, "Mauricio Ruffy": 81, "Sean O'Malley": 80}

print("=== MMA predictions (our model vs Polymarket) ===\n")
for f1, f2 in FIGHTS:
    p1 = predict(f1, f2)
    print(f"{f1} vs {f2}")
    print(f"  our model:  {f1} {p1:.0%}  /  {f2} {1-p1:.0%}")
    fav = f2 if f2 in MARKET else f1
    print(f"  Polymarket: {fav} {MARKET.get(fav)}%\n")
