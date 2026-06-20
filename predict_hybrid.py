"""Hybrid: Elo decides the favourite, Poisson gives the goal markets.

Fixed calibration:
  1. The win/draw/loss is anchored to Elo BY CONSTRUCTION — we solve for the goal
     supremacy that reproduces the Elo expected score, so the hybrid can't over-favour
     the favourite (it matches the direct-Elo model).
  2. Dixon-Coles τ correction bumps up low-score draws (independent Poisson under-counts them).
No home advantage (World Cup is neutral).
"""

import math
from collections import defaultdict
import pandas as pd

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"]).dropna(subset=["home_score"]).sort_values("date")
recent = raw[raw.date.dt.year >= 2018]
T = (recent.home_score.sum() + recent.away_score.sum()) / len(recent)   # current avg total goals
RHO = -0.13   # Dixon-Coles draw correction

# Elo on the FULL history (matches the main model: Germany ~1946, gap ~161)
rating = defaultdict(lambda: 1500.0)
for _, r in raw.iterrows():
    h, a = r.home_team, r.away_team; rh, ra = rating[h], rating[a]
    e = 1 / (1 + 10 ** ((ra - rh) / 400)); s = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    rating[h] = rh + 30 * (s - e); rating[a] = ra + 30 * ((1 - s) - (1 - e))


def pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def tau(i, j, lh, la):  # Dixon-Coles low-score correction
    if i == 0 and j == 0: return 1 - lh * la * RHO
    if i == 0 and j == 1: return 1 + lh * RHO
    if i == 1 and j == 0: return 1 + la * RHO
    if i == 1 and j == 1: return 1 - RHO
    return 1.0


def grid(lh, la):
    N = 9
    P = [[pois(i, lh) * pois(j, la) * tau(i, j, lh, la) for j in range(N)] for i in range(N)]
    tot = sum(sum(row) for row in P)
    return [[x / tot for x in row] for row in P]


def wdl(P):
    N = len(P)
    hw = sum(P[i][j] for i in range(N) for j in range(N) if i > j)
    dr = sum(P[i][j] for i in range(N) for j in range(N) if i == j)
    return hw, dr, 1 - hw - dr


def predict(home, away):
    elo_E = 1 / (1 + 10 ** ((rating[away] - rating[home]) / 400))  # Elo expected score
    # solve supremacy S so the Poisson win+0.5*draw matches Elo's expected score
    lo, hi = -T, T
    for _ in range(40):
        S = (lo + hi) / 2
        P = grid((T + S) / 2, (T - S) / 2)
        hw, dr, aw = wdl(P)
        if hw + 0.5 * dr < elo_E: lo = S
        else: hi = S
    lh, la = (T + S) / 2, (T - S) / 2
    P = grid(lh, la); hw, dr, aw = wdl(P)
    N = len(P)
    over25 = sum(P[i][j] for i in range(N) for j in range(N) if i + j > 2.5)
    btts = sum(P[i][j] for i in range(N) for j in range(N) if i >= 1 and j >= 1)
    print(f"Elo: {home} {rating[home]:.0f} vs {away} {rating[away]:.0f}  (expected score {elo_E:.0%})")
    print(f"expected goals: {home} {lh:.2f}  {away} {la:.2f}")
    print(f"\n  W/D/L:       {home} {hw:.0%} / draw {dr:.0%} / {away} {aw:.0%}")
    print(f"  over 2.5:    {over25:.0%}   ·   both score: {btts:.0%}")
    scores = sorted(((P[i][j], i, j) for i in range(5) for j in range(5)), reverse=True)[:4]
    print("  top scores:  " + ", ".join(f"{i}-{j} ({p:.0%})" for p, i, j in scores))


print(f"avg total goals: {T:.2f}\n")
predict("Germany", "Ivory Coast")
print("\n  gate-check → should match direct Elo model (61/23/16) and market (67/20/14)")
