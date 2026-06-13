"""Live prediction for Brazil vs Morocco.

The backtest (train.py) already estimated accuracy. Here we train on the
WHOLE dataset (maximum information) and predict tonight's match.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["home_win_rate", "home_gf", "home_ga",
            "away_win_rate", "away_gf", "away_ga", "is_neutral"]

# ── train on the whole prepared dataset ─────────────────────────────────
train = pd.read_csv("data/training_set.csv")
scaler = StandardScaler()
X = scaler.fit_transform(train[FEATURES])
model = LogisticRegression().fit(X, train["home_won"])

# ── a team's current form from its last 10 played matches ───────────────
raw = pd.read_csv("data/international_results.csv", parse_dates=["date"])
raw = raw.dropna(subset=["home_score"]).sort_values("date")


def current_form(team, window=10):
    # all of the team's matches (home or away)
    m = raw[(raw.home_team == team) | (raw.away_team == team)].tail(window)
    gf, ga, won = [], [], []
    for _, r in m.iterrows():
        is_home = r.home_team == team
        my = r.home_score if is_home else r.away_score
        opp = r.away_score if is_home else r.home_score
        gf.append(my); ga.append(opp); won.append(1 if my > opp else 0)
    n = len(m)
    return sum(won) / n, sum(gf) / n, sum(ga) / n


bra_wr, bra_gf, bra_ga = current_form("Brazil")
mar_wr, mar_gf, mar_ga = current_form("Morocco")
print(f"Brazil form (last 10):  win_rate={bra_wr:.2f}, scores={bra_gf:.2f}, concedes={bra_ga:.2f}")
print(f"Morocco form (last 10): win_rate={mar_wr:.2f}, scores={mar_gf:.2f}, concedes={mar_ga:.2f}")

# ── match row: Brazil = "home" by schedule, neutral ground (World Cup) ──
row = pd.DataFrame([{
    "home_win_rate": bra_wr, "home_gf": bra_gf, "home_ga": bra_ga,
    "away_win_rate": mar_wr, "away_gf": mar_gf, "away_ga": mar_ga,
    "is_neutral": 1,
}])[FEATURES]

# scale with the SAME scaler and predict the probability
row_s = scaler.transform(row)
proba = model.predict_proba(row_s)[0]  # [P(not win), P(win)]

print("\n" + "=" * 42)
print("  PREDICTION: Brazil vs Morocco")
print("=" * 42)
print(f"  P(Brazil wins)      = {proba[1]:.1%}")
print(f"  P(Brazil does NOT)  = {proba[0]:.1%}")
print("  (does NOT = draw or loss)")
