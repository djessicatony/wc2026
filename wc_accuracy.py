"""Out-of-sample accuracy report over EVERY played World Cup 2026 match.

Unlike predict_wc_backtest.py (which only scores the 32 matches we have
Polymarket odds for), this scores all played WC matches — a bigger, honest
sample. Leakage-safe: model trains only on pre-WC matches; each match's Elo
and form use only earlier games.

Focus: winner accuracy, draw behaviour (the hard class), and probability
quality (log loss / Brier) — not betting.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss

WC_START = pd.Timestamp("2026-06-11")
FEATURES = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf",
            "away_ga", "is_neutral", "elo_dif"]

# ── pre-match Elo over the full match graph (leakage-safe: updated AFTER) ─
raw = (pd.read_csv("data/international_results.csv", parse_dates=["date"])
       .dropna(subset=["home_score"]).sort_values("date"))
rating = defaultdict(lambda: 1500.0)
rows = []
for _, r in raw.iterrows():
    h, a = r.home_team, r.away_team
    rh, ra = rating[h], rating[a]
    rows.append({"date": r.date, "home_team": h, "away_team": a, "elo_dif": rh - ra})
    e = 1 / (1 + 10 ** ((ra - rh) / 400))
    s = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    rating[h] = rh + 30 * (s - e)
    rating[a] = ra + 30 * ((1 - s) - (1 - e))
elo = pd.DataFrame(rows)

df = pd.read_csv("data/training_set.csv", parse_dates=["date"])
df = df.merge(elo, on=["date", "home_team", "away_team"]).merge(
    raw[["date", "home_team", "away_team", "home_score", "away_score"]],
    on=["date", "home_team", "away_team"])
df["result"] = np.where(df.home_score > df.away_score, "home",
                np.where(df.home_score < df.away_score, "away", "draw"))

train = df[df.date < WC_START].dropna(subset=FEATURES + ["result"])
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(train[FEATURES], train["result"])
classes = list(model.classes_)  # ['away','draw','home']

# ── every played WC match (this is the World Cup we are grading) ─────────
wc = df[df.date >= WC_START].dropna(subset=FEATURES + ["result"]).copy()
probs = model.predict_proba(wc[FEATURES])
for i, c in enumerate(classes):
    wc[f"p_{c}"] = probs[:, i]
wc["pick"] = [classes[i] for i in probs.argmax(1)]

n = len(wc)
print(f"=== Out-of-sample over ALL {n} played WC 2026 matches ===\n")

# winner accuracy
acc = (wc.pick == wc.result).mean()
print(f"model winner accuracy : {acc:.1%}  ({(wc.pick==wc.result).sum()}/{n})")

# baseline: always pick the higher-Elo side (home favoured if elo_dif>0)
base_pick = np.where(wc.elo_dif >= 0, "home", "away")
print(f"elo-favourite baseline: {(base_pick==wc.result).mean():.1%}")

# probability quality
y_idx = wc.result.map({c: i for i, c in enumerate(classes)}).values
ll = log_loss(y_idx, wc[[f"p_{c}" for c in classes]].values, labels=range(len(classes)))
print(f"log loss              : {ll:.3f}   (lower better; 3-way coin flip = 1.099)")

# ── DRAWS: the hard class ───────────────────────────────────────────────
print("\n--- draws ---")
actual_draws = (wc.result == "draw").sum()
mean_p_draw = wc.p_draw.mean()
print(f"actual draw rate      : {actual_draws/n:.1%}  ({actual_draws}/{n})")
print(f"model mean P(draw)    : {mean_p_draw:.1%}   <- what the model expected on average")
picked_draw = (wc.pick == "draw").sum()
print(f"times model PICKED draw (argmax): {picked_draw}")
# Brier for the draw class specifically (how well calibrated on draw prob)
brier_draw = brier_score_loss((wc.result == "draw").astype(int), wc.p_draw)
print(f"draw Brier score      : {brier_draw:.3f}  (lower better)")
# recall: of the actual draws, how much probability mass did we put on draw?
print(f"mean P(draw) on the {actual_draws} matches that ACTUALLY drew: "
      f"{wc.loc[wc.result=='draw','p_draw'].mean():.1%}")
print(f"mean P(draw) on the {n-actual_draws} non-draws                : "
      f"{wc.loc[wc.result!='draw','p_draw'].mean():.1%}")

# ── confusion-ish breakdown ─────────────────────────────────────────────
print("\n--- predicted pick vs actual ---")
print(pd.crosstab(wc.pick, wc.result, margins=True))

# ── per-match table ─────────────────────────────────────────────────────
print("\n--- every match (model probs | pick | actual) ---")
show = wc.sort_values("date")[["date", "home_team", "away_team",
                               "p_home", "p_draw", "p_away", "pick", "result"]]
for _, r in show.iterrows():
    hit = "OK " if r.pick == r.result else "  x"
    print(f"{hit} {r.date.date()} {r.home_team[:14]:14} {r.away_team[:14]:14} "
          f"{r.p_home:.2f}/{r.p_draw:.2f}/{r.p_away:.2f}  pick={r.pick:4} act={r.result}")
