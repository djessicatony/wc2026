"""Hybrid: Elo sets the expected goals (λ), Poisson turns them into a full
distribution (W/D/L, totals, both-to-score, scorelines).

Why: Elo propagates strength across the whole match graph (fixes the cross-
confederation problem that broke naive Poisson). Poisson is kept only for what
it's good at — converting expected goals into a probability over outcomes.
We learn the Elo→goals mapping from history: fit λ_home, λ_away vs elo_dif.
"""

import math
from collections import defaultdict
import numpy as np
import pandas as pd

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"]).dropna(subset=["home_score"]).sort_values("date")

# ── pre-match Elo for every match ───────────────────────────────────────
rating = defaultdict(lambda: 1500.0)
elo_dif, ratings_final = [], None
for _, r in raw.iterrows():
    h, a = r.home_team, r.away_team
    rh, ra = rating[h], rating[a]
    elo_dif.append(rh - ra)
    e = 1 / (1 + 10 ** ((ra - rh) / 400)); s = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    rating[h] = rh + 30 * (s - e); rating[a] = ra + 30 * ((1 - s) - (1 - e))
raw["elo_dif"] = elo_dif

# ── learn Elo → expected goals (the mapping that anchors strength) ──────
# home goals and away goals as linear functions of the Elo gap
bh, ah = np.polyfit(raw.elo_dif, raw.home_score, 1)   # λ_home = ah + bh*elo_dif
ba, aa = np.polyfit(raw.elo_dif, raw.away_score, 1)   # λ_away = aa + ba*elo_dif
print(f"learned: λ_home = {ah:.2f} + {bh:.5f}·Δelo,   λ_away = {aa:.2f} + {ba:.5f}·Δelo\n")


def poisson_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def predict(home, away):
    d = rating[home] - rating[away]
    lam_h = max(0.15, ah + bh * d)
    lam_a = max(0.15, aa + ba * d)
    print(f"Elo: {home} {rating[home]:.0f}  vs  {away} {rating[away]:.0f}   (Δ={d:.0f})")
    print(f"expected goals (from Elo):  {home} λ={lam_h:.2f}   {away} λ={lam_a:.2f}")

    N = 9
    P = [[poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a) for j in range(N)] for i in range(N)]
    hw = sum(P[i][j] for i in range(N) for j in range(N) if i > j)
    dr = sum(P[i][j] for i in range(N) for j in range(N) if i == j)
    aw = 1 - hw - dr
    over25 = sum(P[i][j] for i in range(N) for j in range(N) if i + j > 2.5)
    btts = sum(P[i][j] for i in range(N) for j in range(N) if i >= 1 and j >= 1)
    print(f"\n  W/D/L:        {home} {hw:.0%} / draw {dr:.0%} / {away} {aw:.0%}")
    print(f"  over 2.5:     {over25:.0%}   under: {1-over25:.0%}")
    print(f"  both score:   {btts:.0%}")
    scores = sorted(((P[i][j], i, j) for i in range(5) for j in range(5)), reverse=True)[:4]
    print("  top scores:   " + ", ".join(f"{i}-{j} ({p:.0%})" for p, i, j in scores))


predict("Germany", "Ivory Coast")
print("\n  compare → naive Poisson: 37/26/37 (broken) | Elo classifier: 61/23/16 | market: 67/20/14")
