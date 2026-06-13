"""Train the model + honest backtest.

Logistic regression lives here. But first, two anti-leakage steps:
split by date, and scale using train statistics only.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/training_set.csv", parse_dates=["date"])
FEATURES = ["home_win_rate", "home_gf", "home_ga",
            "away_win_rate", "away_gf", "away_ga", "is_neutral"]

# ── STEP 5: split by DATE, not randomly ─────────────────────────────────
# Train on old matches, test on new ones — mirrors reality (predict the
# future from the past). A random split would let the model "see the
# future" -> a dishonest estimate.
df = df.sort_values("date")
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
print(f"train: {len(train)} matches (until {train['date'].max().date()})")
print(f"test:  {len(test)} matches (from {test['date'].min().date()})")

X_train, y_train = train[FEATURES], train["home_won"]
X_test, y_test = test[FEATURES], test["home_won"]

# ── STEP 6: scaling (StandardScaler) ────────────────────────────────────
# Fit on train ONLY: the scaler learns mean/spread from train and applies
# to both. Fitting on all data would leak test statistics into train.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # learn + apply
X_test_s = scaler.transform(X_test)        # apply only

# ── STEP 7: LOGISTIC REGRESSION — here it is, 2 lines ───────────────────
model = LogisticRegression()
model.fit(X_train_s, y_train)

# ── STEP 8: evaluate on test + compare to baselines ─────────────────────
pred = model.predict(X_test_s)
acc = accuracy_score(y_test, pred)

# baseline 1: always say "home team won"
base_home = accuracy_score(y_test, [1] * len(y_test))
# baseline 2: the team with the higher win rate wins (naive favorite)
fav = (X_test["home_win_rate"] > X_test["away_win_rate"]).astype(int)
base_fav = accuracy_score(y_test, fav)

print(f"\n=== TEST ACCURACY ===")
print(f"baseline 'always home':     {base_home:.3f}")
print(f"baseline 'higher win_rate': {base_fav:.3f}")
print(f"our model:                  {acc:.3f}")

# ── learned weights (the 8 numbers fit found) ───────────────────────────
print(f"\n=== WHAT THE MODEL LEARNED (weights) ===")
for name, w in zip(FEATURES, model.coef_[0]):
    arrow = "up to win" if w > 0 else "down from win"
    print(f"  {name:16} {w:+.3f}  {arrow}")
print(f"  {'bias (b)':16} {model.intercept_[0]:+.3f}")
