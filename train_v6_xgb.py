"""v6: now that we have a strong feature (Elo), does XGBoost beat logreg?

Earlier XGBoost tied/lost on thin v1 features. Re-test on form + Elo,
both binary and 3-way, same date split. Measure, don't guess.
"""

from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"])
raw = raw.dropna(subset=["home_score"]).sort_values("date")

# simple Elo (leakage-safe)
ratings = defaultdict(lambda: 1500.0)
rows = []
for _, r in raw.iterrows():
    h, a = r.home_team, r.away_team
    rh, ra = ratings[h], ratings[a]
    rows.append({"date": r.date, "home_team": h, "away_team": a, "home_elo": rh, "away_elo": ra})
    exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
    sh = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    ratings[h] = rh + 30 * (sh - exp_h)
    ratings[a] = ra + 30 * ((1 - sh) - (1 - exp_h))
elo = pd.DataFrame(rows)

keys = ["date", "home_team", "away_team"]
df = pd.read_csv("data/training_set.csv", parse_dates=["date"])
df = df.merge(raw[keys + ["home_score", "away_score"]], on=keys).merge(elo, on=keys).dropna().sort_values("date")
df["result"] = (df.home_score > df.away_score).astype(int) * 2 + (df.home_score == df.away_score).astype(int)

F = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf", "away_ga", "is_neutral", "home_elo", "away_elo"]
cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]


def compare(target, label):
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=1000).fit(sc.fit_transform(tr[F]), tr[target])
    lr_acc = accuracy_score(te[target], lr.predict(sc.transform(te[F])))
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, eval_metric="logloss")
    xgb.fit(tr[F], tr[target])  # trees don't need scaling
    xgb_acc = accuracy_score(te[target], xgb.predict(te[F]))
    print(f"{label:20} logreg {lr_acc:.3f}   XGBoost {xgb_acc:.3f}")


print("=== form + Elo, logreg vs XGBoost ===")
compare("home_won", "binary (win/not):")
compare("result", "3-way (W/D/L):")
