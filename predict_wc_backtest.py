"""Out-of-sample test: predict every collected WC match leakage-safe, then run
Kelly against the closing Polymarket odds we pulled.

Leakage-safe: the model trains ONLY on matches before the World Cup started,
and each match's features (form, Elo) use only earlier matches.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

WC_START = pd.Timestamp("2026-06-11")
FEATURES = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf", "away_ga", "is_neutral", "elo_dif"]

# ── leakage-safe form is already in training_set.csv; add pre-match Elo ──
raw = pd.read_csv("data/international_results.csv", parse_dates=["date"]).dropna(subset=["home_score"]).sort_values("date")
rating = defaultdict(lambda: 1500.0); elo_rows = []
for _, r in raw.iterrows():
    h, a = r.home_team, r.away_team; rh, ra = rating[h], rating[a]
    elo_rows.append({"date": r.date, "home_team": h, "away_team": a, "elo_dif": rh - ra})
    e = 1 / (1 + 10 ** ((ra - rh) / 400)); s = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    rating[h] = rh + 30 * (s - e); rating[a] = ra + 30 * ((1 - s) - (1 - e))
elo = pd.DataFrame(elo_rows)

df = pd.read_csv("data/training_set.csv", parse_dates=["date"])
df = df.merge(elo, on=["date", "home_team", "away_team"]).merge(
    raw[["date", "home_team", "away_team", "home_score", "away_score"]], on=["date", "home_team", "away_team"])
df["result"] = np.where(df.home_score > df.away_score, "home", np.where(df.home_score < df.away_score, "away", "draw"))

# train multinomial ONLY on pre-WC matches
train = df[df.date < WC_START].dropna(subset=FEATURES + ["result"])
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(train[FEATURES], train["result"])
classes = list(model.classes_)  # ['away','draw','home']

# ── attach model probs to the collected Polymarket matches ──────────────
mkt = pd.read_csv("data/wc_polymarket.csv", parse_dates=["date"])
mkt["date"] = mkt["date"].dt.normalize()
feat = df[["date", "home", "away"] if False else ["date", "home_team", "away_team"] + FEATURES].rename(
    columns={"home_team": "home", "away_team": "away"})
m = mkt.merge(feat, on=["date", "home", "away"], how="left").dropna(subset=FEATURES)
probs = model.predict_proba(m[FEATURES])
for i, c in enumerate(classes):
    m[f"model_{c}"] = probs[:, i]

print(f"matches with model + market + result: {len(m)}")

# ── Kelly on the real sample ────────────────────────────────────────────
START, K = 100.0, 0.5
def run(label, cap=None, draws_only=False):
    bank = START
    for _, row in m.iterrows():
        sides = [("home", "mkt_home", "model_home"), ("draw", "mkt_draw", "model_draw"), ("away", "mkt_away", "model_away")]
        for side, mc, pc in sides:
            if draws_only and side != "draw":
                continue
            price, p = row[mc], row[pc]
            edge = p - price
            if edge <= 0 or price <= 0 or price >= 1:
                continue
            if cap:
                edge = min(edge, cap)
            stake = K * edge / (1 - price) * bank
            bank += stake * (1 / price - 1) if row["result"] == side else -stake
    roi = (bank - START) / START * 100
    print(f"  {label:24} ${bank:6.2f}  (ROI {roi:+.1f}%)")

print(f"\n=== Kelly on {len(m)} WC matches (out-of-sample) ===")
run("naive Kelly")
run("capped 5%", cap=0.05)
run("draws-only", draws_only=True)

# how often did the model just pick the right winner?
m["model_pick"] = m[[f"model_{c}" for c in classes]].values.argmax(1)
m["pick"] = [classes[i] for i in m["model_pick"]]
print(f"\nmodel winner accuracy: {(m.pick == m.result).mean():.1%}")
print(f"market favourite accuracy: {(m[['mkt_home','mkt_draw','mkt_away']].values.argmax(1) == m.result.map({'home':0,'draw':1,'away':2})).mean():.1%}")
