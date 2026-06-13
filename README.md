# ⚽ Match Predictor

A football match-outcome predictor built as a hands-on intro to classical ML.
Predicts whether a national team wins, from recent form, using logistic
regression — then stress-tests the result against XGBoost and a real prediction
market.

> First ML project. Goal: understand every line, not just run magic.

## What it does

Predicts **P(home team wins)** for an international match from each team's recent
form (win rate, goals for/against) plus whether the match is on neutral ground.

Worked example: **Brazil vs Morocco (World Cup 2026)** → model says Brazil 29.1%.

## Pipeline

```
international results CSV  (49k matches, 1872–2026)
        │  build_dataset.py  — clean, leakage-safe rolling form, assemble rows
        ▼
training_set.csv  (25k matches since 2000, 7 features → answer)
        │  train.py  — logistic regression + chronological backtest
        ▼
65% accuracy on held-out matches  (beats "always home" 48%, "higher form" 63%)
        │  predict.py  — train on all data, predict one match
        ▼
Brazil 29.1%
```

## Files (the logic)

| file | role |
|---|---|
| `fetch_matches.py` | football-data.org API client (team IDs; rate-limit aware) |
| `build_dataset.py` | feature engineering: raw matches → leakage-safe form features |
| `train.py` | train logistic regression, backtest by date split |
| `predict.py` | live prediction for one match |
| `compare_models.py` | logistic regression vs XGBoost (same data) |
| `build_statsbomb.py` | v2: aggregate StatsBomb events into rich features (xG, possession) |
| `train_statsbomb.py` | v2: train/compare on rich features |

## Key results & lessons

- **65%** backtest accuracy (logistic regression, 25k matches). See `OVERVIEW.md`.
- **XGBoost did not beat logistic regression** on these features (`compare_models.py`)
  — the data, not the model, is the ceiling.
- **Rich StatsBomb features (xG) underperformed** simple ones because they exist
  for far fewer matches; XGBoost overfit the small set. See `OVERVIEW.md` v2.
- **Model vs prediction market**: we said Brazil 29%, Polymarket said 59% —
  a calibration failure traced to a missing "strength of schedule" feature.
  See `CALIBRATION.md`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas scikit-learn xgboost requests python-dotenv

# data is gitignored — rebuild it:
curl -sSL https://raw.githubusercontent.com/martj42/international_results/master/results.csv \
  -o data/international_results.csv
python build_dataset.py   # → data/training_set.csv
python train.py           # backtest
python predict.py         # predict the match
```

## Data sources

- [martj42/international_results](https://github.com/martj42/international_results) — match results (main)
- [StatsBomb open data](https://github.com/statsbomb/open-data) — event data for v2 (xG, possession)
- football-data.org — team IDs (free tier blocks national-team history)
