"""Turn the raw match CSV into a training dataset (features -> answer).

Steps:
  1. load & clean
  2a. explode each match into 2 team-perspective rows
  2b. rolling form over previous matches (leakage-safe)
  3. assemble one row per match (home form + away form + answer)
  4. drop rows without form, save artifact
"""

import pandas as pd

# ── STEP 1: load and clean ──────────────────────────────────────────────
df = pd.read_csv("data/international_results.csv", parse_dates=["date"])
print("raw rows:", len(df))

# 1a. drop unplayed matches (score = NA: future fixtures)
df = df.dropna(subset=["home_score", "away_score"])
print("after dropping unplayed:", len(df))

# 1b. drop ancient matches — keep modern football from 2000 on
df = df[df["date"].dt.year >= 2000]
print("after cutting pre-2000:", len(df))

# 1c. sort by date (needed to compute form "from previous matches")
df = df.sort_values("date").reset_index(drop=True)
print("\ndone. range:", df["date"].min().date(), "…", df["date"].max().date())

# give each match a unique id — used later to reassemble rows
df["match_id"] = df.index

# ── STEP 2a: explode each match into 2 rows (each team's perspective) ────
# home perspective: its goals = home_score, conceded = away_score
home_view = pd.DataFrame({
    "match_id": df["match_id"],
    "date": df["date"],
    "team": df["home_team"],
    "goals_for": df["home_score"],
    "goals_against": df["away_score"],
    "won": (df["home_score"] > df["away_score"]).astype(int),  # True/False -> 1/0
})
# away perspective: mirrored — its goals = away_score
away_view = pd.DataFrame({
    "match_id": df["match_id"],
    "date": df["date"],
    "team": df["away_team"],
    "goals_for": df["away_score"],
    "goals_against": df["home_score"],
    "won": (df["away_score"] > df["home_score"]).astype(int),
})

# stack both vertically -> each team gets its own chronology
long = pd.concat([home_view, away_view], ignore_index=True)
long = long.sort_values(["team", "date"]).reset_index(drop=True)
print("\nmatches:", len(df), "-> team-perspective rows:", len(long), "(exactly x2)")

# ── STEP 2b: form over the previous 10 matches (no leakage) ──────────────
WINDOW = 10
grp = long.groupby("team")  # compute within each team separately, never mixing


def form(col):
    # .shift(1): move down one row -> current match EXCLUDED (anti-leakage)
    # .rolling(WINDOW): window of the 10 previous rows
    # .mean(): average over them
    return grp[col].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean()
    )


long["form_win_rate"] = form("won")
long["form_gf"] = form("goals_for")
long["form_ga"] = form("goals_against")

# ── STEP 3: one row per match (home form + away form + answer) ───────────
# home form: join on (match_id, home_team)
home_forms = long[["match_id", "team", "form_win_rate", "form_gf", "form_ga"]].rename(
    columns={"team": "home_team", "form_win_rate": "home_win_rate",
             "form_gf": "home_gf", "form_ga": "home_ga"})
df = df.merge(home_forms, on=["match_id", "home_team"], how="left")

# away form: join on (match_id, away_team)
away_forms = long[["match_id", "team", "form_win_rate", "form_gf", "form_ga"]].rename(
    columns={"team": "away_team", "form_win_rate": "away_win_rate",
             "form_gf": "away_gf", "form_ga": "away_ga"})
df = df.merge(away_forms, on=["match_id", "away_team"], how="left")

# answer: did the home team win (the thing we predict)
df["home_won"] = (df["home_score"] > df["away_score"]).astype(int)
# neutral ground as a number
df["is_neutral"] = df["neutral"].astype(int)

# ── STEP 4: drop rows without form (teams' first matches -> NaN) ─────────
FEATURE_COLS = ["home_win_rate", "home_gf", "home_ga",
                "away_win_rate", "away_gf", "away_ga", "is_neutral"]
before = len(df)
df = df.dropna(subset=FEATURE_COLS)
print(f"\ndropped {before - len(df)} rows without form -> {len(df)} training rows left")

# final matrices: X = features, y = answer
X = df[FEATURE_COLS]
y = df["home_won"]
print(f"\nX: {X.shape} (rows, features)   y: {y.shape}")

# save the finished dataset (features + answer + date) as a training artifact
out = df[["date", "home_team", "away_team"] + FEATURE_COLS + ["home_won"]]
out.to_csv("data/training_set.csv", index=False)
print("\nsaved data/training_set.csv —", len(out), "rows, ready to train")
