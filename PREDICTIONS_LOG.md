# Predictions log (locked before kickoff)

Model: form + Elo features. 3-way via multinomial softmax. Neutral ground.
Tracked against Polymarket and against sujar.tech (Instagram analyst running a
StatsBomb + XGBoost model) for an ongoing comparison.

## Brazil vs Morocco — World Cup 2026, June 14

| | logreg | market (Polymarket) | result |
|---|---|---|---|
| Brazil win | 36% | 59% | — |
| Draw | 28% | 26% | **✓ 1–1** |
| Morocco win | 36% | 17% | — |

Outcome: 1–1 draw, xG 1.28–1.24 (even). Model's "even match" read beat the
market's Brazil bias. Full analysis in `POSTMORTEM.md`.

## Netherlands vs Japan — World Cup 2026

Elo: Netherlands 1917, Japan 1911 (near-equal). Japan slightly better recent form.

| | our logreg | sujar.tech | market | result |
|---|---|---|---|---|
| Netherlands win | 35.7% | 53% | 48% | — |
| Draw | 30.2% | 29% | 28% | **✓ 2–2** |
| Japan win | 34.1% | 18% | 26% | — |

Result: 2–2 draw (xG 0.70–0.54, close). Second football match running where the
model's "even" read beat the market's favourite-lean. Netherlands didn't win.

Here the two data models **split**: ours reads an even match, while sujar.tech
(53% Netherlands) sides with the market. We are the contrarian — most bullish
on Japan (34% vs market 26% vs sujar.tech 18%). To be checked after the match.

> Reproduce: `python predict_match.py "Netherlands" "Japan"`
