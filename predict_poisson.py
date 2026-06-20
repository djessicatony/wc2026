"""Poisson goals model — generative: model each team's goal RATE, then let the
Poisson distribution give probabilities for every scoreline → W/D/L, totals, BTTS.

Not a scoreline classifier (exact scores are near-random). We estimate two
numbers (expected goals per side) and derive the whole outcome distribution.
"""

import math
from collections import defaultdict
import pandas as pd

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"]).dropna(subset=["home_score"]).sort_values("date")
raw = raw[raw.date.dt.year >= 2018]  # recent era for current strengths

# league average goals per team per match
LEAGUE_AVG = (raw.home_score.sum() + raw.away_score.sum()) / (2 * len(raw))


def _matches(team, window):
    m = raw[(raw.home_team == team) | (raw.away_team == team)].tail(window)
    out = []
    for _, r in m.iterrows():
        opp = r.away_team if r.home_team == team else r.home_team
        gf = r.home_score if r.home_team == team else r.away_score
        ga = r.away_score if r.home_team == team else r.home_score
        out.append((opp, gf, ga))
    return out


def raw_strength(team, window=12):
    """naive: goals scored/conceded vs league avg (ignores who the opponent was)"""
    ms = _matches(team, window)
    return (sum(g for _, g, _ in ms) / len(ms) / LEAGUE_AVG,
            sum(g for _, _, g in ms) / len(ms) / LEAGUE_AVG)


def strength(team, window=12):
    """opponent-adjusted: a goal vs a strong defence counts more (Dixon-Coles idea)"""
    ms = _matches(team, window)
    att, dfn = [], []
    for opp, gf, ga in ms:
        opp_att, opp_def = raw_strength(opp)            # how good was the opponent?
        att.append(gf / (LEAGUE_AVG * max(opp_def, 0.3)))   # goals vs their defence
        dfn.append(ga / (LEAGUE_AVG * max(opp_att, 0.3)))   # conceded vs their attack
    return sum(att) / len(att), sum(dfn) / len(dfn)


def poisson_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def predict(home, away):
    a_h, d_h = strength(home)
    a_a, d_a = strength(away)
    # expected goals: your attack x opponent's leakiness x league average
    lam_h = LEAGUE_AVG * a_h * d_a
    lam_a = LEAGUE_AVG * a_a * d_h
    print(f"{home}: attack {a_h:.2f}, defence {d_h:.2f}   →  expected goals λ={lam_h:.2f}")
    print(f"{away}: attack {a_a:.2f}, defence {d_a:.2f}   →  expected goals λ={lam_a:.2f}")

    # joint scoreline grid (independent Poisson)
    N = 9
    P = [[poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a) for j in range(N)] for i in range(N)]
    home_w = sum(P[i][j] for i in range(N) for j in range(N) if i > j)
    draw = sum(P[i][j] for i in range(N) for j in range(N) if i == j)
    away_w = sum(P[i][j] for i in range(N) for j in range(N) if i < j)
    over25 = sum(P[i][j] for i in range(N) for j in range(N) if i + j > 2.5)
    btts = sum(P[i][j] for i in range(N) for j in range(N) if i >= 1 and j >= 1)

    print(f"\n  W / D / L:   {home} {home_w:.0%} / draw {draw:.0%} / {away} {away_w:.0%}")
    print(f"  over 2.5:    {over25:.0%}   under 2.5: {1-over25:.0%}")
    print(f"  both score:  {btts:.0%}")
    # most likely scorelines
    scores = sorted(((P[i][j], i, j) for i in range(5) for j in range(5)), reverse=True)[:4]
    print("  top scorelines: " + ", ".join(f"{i}-{j} ({p:.0%})" for p, i, j in scores))


def wdl_only(home, away, strength_fn):
    a_h, d_h = strength_fn(home); a_a, d_a = strength_fn(away)
    lam_h, lam_a = LEAGUE_AVG * a_h * d_a, LEAGUE_AVG * a_a * d_h
    N = 9
    P = [[poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a) for j in range(N)] for i in range(N)]
    hw = sum(P[i][j] for i in range(N) for j in range(N) if i > j)
    dr = sum(P[i][j] for i in range(N) for j in range(N) if i == j)
    return lam_h, lam_a, hw, dr, 1 - hw - dr


print(f"league avg goals/team/match: {LEAGUE_AVG:.2f}\n")
lh, la, hw, dr, aw = wdl_only("Germany", "Ivory Coast", raw_strength)
print(f"NAIVE (no opponent adjustment): λ {lh:.2f}/{la:.2f} → Germany {hw:.0%} / draw {dr:.0%} / Ivory Coast {aw:.0%}")
print("ADJUSTED (goals weighted by opponent strength):")
predict("Germany", "Ivory Coast")
print("\n  market (Polymarket): Germany 67% / draw 20% / Côte d'Ivoire 14%")
