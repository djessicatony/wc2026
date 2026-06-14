"""Sensitivity analysis — how much does the prediction move when we change
assumptions (form window, Elo K-factor)? A robust prediction barely moves;
a fragile one swings. Saves a heatmap per match.
"""

from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sns.set_theme(style="white", context="talk", font_scale=0.8)

raw = pd.read_csv("data/international_results.csv", parse_dates=["date"]).dropna(subset=["home_score"]).sort_values("date")
WINDOWS = [5, 10, 15, 20]
KS = [20, 30, 40]
F = ["home_win_rate", "home_gf", "home_ga", "away_win_rate", "away_gf", "away_ga", "is_neutral", "home_elo", "away_elo"]


def elo_for(k):
    r = defaultdict(lambda: 1500.0); rows = []
    for _, m in raw.iterrows():
        h, a = m.home_team, m.away_team; rh, ra = r[h], r[a]
        rows.append((m.date, h, a, rh, ra))
        e = 1 / (1 + 10 ** ((ra - rh) / 400)); s = 1.0 if m.home_score > m.away_score else (0.5 if m.home_score == m.away_score else 0.0)
        r[h] = rh + k * (s - e); r[a] = ra + k * ((1 - s) - (1 - e))
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_elo", "away_elo"]), r


def form_for(window):
    df = raw[raw.date.dt.year >= 2000].copy(); df["mid"] = range(len(df))
    h = pd.DataFrame({"mid": df.mid, "date": df.date, "team": df.home_team, "gf": df.home_score, "ga": df.away_score, "won": (df.home_score > df.away_score).astype(int)})
    a = pd.DataFrame({"mid": df.mid, "date": df.date, "team": df.away_team, "gf": df.away_score, "ga": df.home_score, "won": (df.away_score > df.home_score).astype(int)})
    lo = pd.concat([h, a]).sort_values(["team", "date"]); g = lo.groupby("team")
    for c in ["won", "gf", "ga"]:
        lo["x_" + c] = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    hf = lo[["mid", "team", "x_won", "x_gf", "x_ga"]].rename(columns={"team": "home_team", "x_won": "home_win_rate", "x_gf": "home_gf", "x_ga": "home_ga"})
    af = lo[["mid", "team", "x_won", "x_gf", "x_ga"]].rename(columns={"team": "away_team", "x_won": "away_win_rate", "x_gf": "away_gf", "x_ga": "away_ga"})
    df = df.merge(hf, on=["mid", "home_team"]).merge(af, on=["mid", "away_team"])
    df["home_won"] = (df.home_score > df.away_score).astype(int); df["is_neutral"] = df.neutral.astype(int)
    return df


def cur_form(team, window):
    m = raw[(raw.home_team == team) | (raw.away_team == team)].tail(window)
    gf = [r.home_score if r.home_team == team else r.away_score for _, r in m.iterrows()]
    ga = [r.away_score if r.home_team == team else r.home_score for _, r in m.iterrows()]
    won = [1 if f > g else 0 for f, g in zip(gf, ga)]
    n = len(m) or 1
    return sum(won) / n, sum(gf) / n, sum(ga) / n


# precompute (3 elo passes, 4 form builds) instead of recomputing per cell
elos = {k: elo_for(k) for k in KS}
forms = {w: form_for(w) for w in WINDOWS}


def grid(home, away):
    g = np.zeros((len(WINDOWS), len(KS)))
    for i, w in enumerate(WINDOWS):
        for j, k in enumerate(KS):
            elo, ratings = elos[k]
            df = forms[w].merge(elo, on=["date", "home_team", "away_team"]).dropna()
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(df[F], df["home_won"])
            hf, af = cur_form(home, w), cur_form(away, w)
            row = pd.DataFrame([[hf[0], hf[1], hf[2], af[0], af[1], af[2], 1, ratings[home], ratings[away]]], columns=F)
            g[i, j] = model.predict_proba(row)[0][1] * 100
    return g


def heatmap(home, away, fname):
    g = grid(home, away)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(g, annot=True, fmt=".1f", cmap="RdYlGn_r", center=50, vmin=20, vmax=70,
                xticklabels=[f"K={k}" for k in KS], yticklabels=[f"window={w}" for w in WINDOWS],
                cbar_kws={"label": f"P({home} win) %"}, ax=ax, linewidths=1, linecolor="white")
    ax.set_title(f"{home} vs {away}: P({home} win)\nacross assumptions (range {g.min():.0f}–{g.max():.0f}%)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"assets/{fname}.png", dpi=140); plt.close(fig)
    print(f"{home} vs {away}: P({home} win) ranges {g.min():.1f}%–{g.max():.1f}% (swing {g.max()-g.min():.1f} pts)")


heatmap("Brazil", "Morocco", "sensitivity_brazil_morocco")
heatmap("Netherlands", "Japan", "sensitivity_netherlands_japan")
