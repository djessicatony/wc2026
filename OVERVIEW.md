# Match Predictor — Brazil vs Morocco (World Cup 2026)

**Deadline with teeth:** match kicks off Sun June 14, 03:00 KZ time. We build before kickoff; reality grades the model after. This replaces the salary predictor as Day 1's classical-ML project (same concepts, better stakes).

## The stack (the three logos from the reel)

- **NumPy** — fast array math in C; the engine underneath everything
- **pandas** — DataFrames: in-memory SQL-ish tables (rows = matches, columns = stats)
- **scikit-learn** — the models + utilities: `LogisticRegression`, `StandardScaler`, `train_test_split`

Python-for-TS-devs cheat sheet: `range(3)` = `0..2` · `append` = `push` · `i**2` = `i ** 2` (same operator) · idiomatic pandas avoids loops entirely — `df['new'] = df['col'] ** 2` applies to the whole column at once (**vectorization**, executed in C).

## The question (framed properly)

"Will Brazil win this match — yes or no?" Binary on purpose: a match has 3 outcomes (win/draw/lose), so we fold draw into "no." Getting the question framing right is half of applied ML.

## The pipeline (= the reel, demystified)

| Step | What it is | Who writes it |
|---|---|---|
| 1. Pull match data from a football API | Historical matches for both teams (API-Football or football-data.org free tier) | Claude scaffolds the client, you pick the teams/window |
| 2. Build a DataFrame | In-memory SQL-ish table via pandas: rows = matches, columns = raw stats | You, guided |
| 3. Feature engineering | Design the input columns: rolling win rate, goals scored/conceded per match, form (last 5), home/neutral flag, opponent strength | **You — this is the core learning** |
| 4. StandardScaler | Normalize columns to the same scale so big-numbered stats don't dominate the learned weights | You (2 lines, but explain why) |
| 5. Logistic regression | Function with learned weights, output squashed to 0–1 = P(Brazil wins) | You (sklearn, ~5 lines) |
| 6. predict_proba | Get the probability for Sunday's match | You |

## The part the reel skipped — and it's the most important part

**How do we know the model isn't garbage before Sunday?** Backtest: hold out the most recent ~20% of historical matches, train only on the older ones, then score predictions on the held-out recent matches. Two numbers:

- **Accuracy** — % of held-out matches predicted correctly
- **Baseline to beat** — "always predict the stronger team wins." If the model can't beat that dumb rule, the features aren't earning their keep.

This held-out scoring is your first **eval** — same concept that runs the whole RAG project later, and the #1 thing the reel didn't mention.

⚠️ **Leakage trap (will come up while building):** every feature for a match must be computed *only from matches before it*. If "win rate" includes the match you're predicting, the model cheats during backtest and faceplants in reality. The #1 beginner bug in this exact project.

## Honest expectations

Football is noisy — single matches are close to coin flips even for bookmakers (whose odds are an excellent sanity check for our probability). A good outcome is a model slightly better than the naive baseline that produces a *calibrated* probability, plus you understanding every line. If it says 65% Brazil and Morocco wins, the model isn't "wrong" — a 35% event happened. Probabilities are graded over many predictions, not one. (That's also your first taste of why evals > vibes.)

## Concepts unlocked

DataFrames · feature engineering · train/test split · leakage · scaling · logistic regression · probability calibration · baseline comparison · backtesting (= evals)

## Data sources

- **Active:** `data/international_results.csv` (martj42/international_results) — 49k+ national-team results since 1872. Wide, shallow: date, teams, score, tournament, neutral flag.
- **v2 candidate:** StatsBomb open data (`statsbomb/open-data` + `statsbombpy`) — event-level data (passes, shots, xG, possession) for select competitions (WC 2018/2022, Euros). Deep, narrow. Unlocks composite features like "win rate vs dominant attacking teams" (define via possession/xG). Not needed for v1.
- **Rejected:** football-data.org free tier — TIER_ONE blocks national-team history (403).

## v2 результаты (StatsBomb богатые фичи) — эксперимент проведён

Гипотеза: богатые фичи (xG, владение, точность) + XGBoost побьют простой v1. **Опровергнута замером.**

| | логрегрессия | XGBoost |
|---|---|---|
| v1 (простые фичи, 25080 матчей) | 0.645 | 0.641 |
| v2 (богатые фичи, 268 матчей) | 0.632 | 0.573 |

Выводы:
1. **Богатые фичи не побили простые** — объём строк (25080 vs 268) перевесил богатство сигнала.
2. **XGBoost на 268 матчах переобучился** (0.573, хуже логрегрессии) — 14 фич × мало строк = зубрёжка.
3. **Размен подтверждён:** богатые фичи покупаются ценой числа строк, и здесь объём победил.
4. **Оговорка:** тест-сеты v1/v2 разные (v2 — только турнирные матчи равных сборных, труднее предсказать), так что сравнение не идеально чистое. Чистое сравнение = мерить на одних матчах.

Главный урок: модель не узкое место — данные/фичи задают потолок; сложная модель на малых данных переобучается.

## Status

- [ ] Pick data API + get key
- [ ] Pull historical matches → DataFrame
- [ ] Features (leakage-safe)
- [ ] Split + train + backtest vs baseline
- [ ] Predict Brazil vs Morocco, write the number down BEFORE kickoff
- [ ] Sunday: compare vs reality + bookmaker odds, write `explainers/01-classical-ml.md`
