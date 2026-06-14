# ⚽ World Cup 2026 Match Predictor

A side project: a football model that estimates **win / draw / loss
probabilities** for any international fixture, benchmarked live against the
Polymarket prediction market.

The idea was simple — build a clean results-based model and see how it stacks
up against a $12M betting market on real World Cup matches.

> **Brazil vs Morocco:** the model called an even match (36/28/36) while the
> market heavily favoured Brazil (59/26/17). It finished **1–1**, with
> near-identical expected goals (1.28 vs 1.24).

<p align="center">
  <img src="assets/prediction_vs_market.png" width="80%">
</p>

---

## What it does

Given two national teams, it predicts the outcome from each team's **recent
form** (win rate, goals scored/conceded), an **Elo strength rating**, and
whether the match is on neutral ground.

```bash
python predict_match.py "Netherlands" "Japan"
```
```
3-WAY            logreg    XGBoost
  Netherlands     35.7%     35.3%
  Draw            30.2%     28.6%
  Japan           34.1%     36.1%
```

## How it works

```
49k international results (1872–2026)
        │  build_dataset.py — clean, leakage-safe rolling form
        ▼
25k matches since 2000  (form features + Elo + outcome)
        │  train.py / train_v3_elo.py — logistic regression + backtest
        ▼
71% accuracy on held-out matches  (chronological split, no leakage)
        │  predict_match.py — train on all data, predict one fixture
        ▼
win / draw / loss probabilities
```

Two rules keep the backtest honest:
- **Form uses only matches *before* each game** (`shift(1)` + rolling window).
- **The model is tested on *future* matches** (split by date, never shuffled).

## Model development

Iterations and what moved the accuracy, measured on the same held-out backtest:

<p align="center">
  <img src="assets/accuracy_by_version.png" width="80%">
</p>

| Version | Change | Backtest |
|---|---|---|
| v1 | form only | 65.0% |
| v2 | XGBoost + rich StatsBomb features (xG, possession), few matches | 57.3% |
| **v3** | **+ Elo rating (opponent strength)** | **71.2%** |
| v6 | XGBoost on form + Elo | 70.7% |
| v7 | + match importance | 71.1% |

The Elo feature added +6 points; switching to XGBoost added nothing across
three separate tests. On this problem, **features matter more than model
choice.**

## Calibration

Probabilities are calibrated — when the model says 30%, the home side wins
≈29% of the time (`train_v8_calibration.py`), so the percentages mean what
they say.

## Benchmark vs Polymarket

| Match | Model (W/D/L) | Polymarket | Result |
|---|---|---|---|
| Brazil – Morocco | 36 / 28 / 36 | 59 / 26 / 17 | **1–1** (xG 1.28–1.24) |
| Netherlands – Japan | 36 / 30 / 34 | 48 / 28 / 26 | *pending* |

A recurring pattern: the model agrees with the market on the **draw** but
rates the **underdog** higher — it reads strength from results, where the
market leans on reputation. Full write-up in [`POSTMORTEM.md`](POSTMORTEM.md)
and [`CALIBRATION.md`](CALIBRATION.md).

<p align="center">
  <img src="assets/polymarket_brazil_morocco.png" width="48%">
  <img src="assets/polymarket_netherlands_japan.png" width="48%">
</p>

## Files

| file | role |
|---|---|
| `build_dataset.py` | feature engineering: raw matches → leakage-safe form features |
| `train.py` | logistic regression + backtest by date split |
| `train_v3_elo.py` | Elo opponent-strength feature |
| `train_v5_3way.py` | 3-way prediction via multinomial softmax |
| `compare_models.py`, `train_v6_xgb.py` | logistic regression vs XGBoost |
| `train_v8_calibration.py` | probability calibration |
| `predict_match.py` | **predict any fixture with all models** |
| `make_viz.py` | generate the charts above |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas scikit-learn xgboost matplotlib requests python-dotenv

mkdir -p data
curl -sSL https://raw.githubusercontent.com/martj42/international_results/master/results.csv \
  -o data/international_results.csv
python build_dataset.py
python predict_match.py "Brazil" "Morocco"
```

## Data sources

- [martj42/international_results](https://github.com/martj42/international_results) — match results
- [StatsBomb open data](https://github.com/statsbomb/open-data) — event data (xG, possession)
- football-data.org — team IDs
