"""v5: 3-way prediction (home win / draw / away win) = multiclass.

Binary logreg used one score -> sigmoid -> P(win). Multiclass computes one
score per outcome -> softmax -> 3 probabilities that sum to 1.
Same model family, 3-class answer. Built on the best feature base (form + Elo).
"""

from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"])
raw = raw.dropna(subset=["home_score"]).sort_values("date")

# ── simple Elo (leakage-safe), same as v3 ───────────────────────────────
ratings = defaultdict(lambda: 1500.0)
elo_rows = []
for _, r in raw.iterrows():
    h, a = r.home_team, r.away_team
    rh, ra = ratings[h], ratings[a]
    elo_rows.append({"date": r.date, "home_team": h, "away_team": a, "home_elo": rh, "away_elo": ra})
    exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
    sh = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    ratings[h] = rh + 30 * (sh - exp_h)
    ratings[a] = ra + 30 * ((1 - sh) - (1 - exp_h))
elo = pd.DataFrame(elo_rows)

# ── 3-class label: 0 = away win, 1 = draw, 2 = home win ─────────────────
keys = ["date", "home_team", "away_team"]
scores = raw[keys + ["home_score", "away_score"]]
df = pd.read_csv("data/training_set.csv", parse_dates=["date"])
df = df.merge(scores, on=keys).merge(elo, on=keys).dropna().sort_values("date")
df["result"] = (df["home_score"] > df["away_score"]).astype(int) * 2 + (df["home_score"] == df["away_score"]).astype(int)

FEATURES = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf",
            "away_ga", "is_neutral", "home_elo", "away_elo"]
cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]

sc = StandardScaler()
model = LogisticRegression(max_iter=1000).fit(sc.fit_transform(tr[FEATURES]), tr["result"])
acc = accuracy_score(te["result"], model.predict(sc.transform(te[FEATURES])))

# baseline: always predict the most common class in train
common = tr["result"].mode()[0]
base = accuracy_score(te["result"], [common] * len(te))

print("=== 3-WAY BACKTEST ===")
print(f"baseline (always most common outcome): {base:.3f}")
print(f"our 3-way model:                       {acc:.3f}")
print("(lower than binary 0.71 — 3 classes is harder, draws especially)")

# ── predict Brazil vs Morocco (3 probabilities) ─────────────────────────
def current_form(team, window=10):
    m = raw[(raw.home_team == team) | (raw.away_team == team)].tail(window)
    gf = [r.home_score if r.home_team == team else r.away_score for _, r in m.iterrows()]
    ga = [r.away_score if r.home_team == team else r.home_score for _, r in m.iterrows()]
    won = [1 if f > g else 0 for f, g in zip(gf, ga)]
    n = len(m)
    return sum(won) / n, sum(gf) / n, sum(ga) / n


bra = current_form("Brazil"); mar = current_form("Morocco")
row = pd.DataFrame([{
    "home_win_rate": bra[0], "home_gf": bra[1], "home_ga": bra[2],
    "away_win_rate": mar[0], "away_gf": mar[1], "away_ga": mar[2],
    "is_neutral": 1, "home_elo": ratings["Brazil"], "away_elo": ratings["Morocco"],
}])[FEATURES]
p = model.predict_proba(sc.transform(row))[0]  # order: [away win, draw, home win]

print("\n=== PREDICTION: Brazil vs Morocco (3-way) ===")
print(f"  Brazil win:  {p[2]:.1%}   (market 59%)")
print(f"  Draw:        {p[1]:.1%}   (market 26%)")
print(f"  Morocco win: {p[0]:.1%}   (market 17%)")
print(f"  sum = {p.sum():.2f}")
