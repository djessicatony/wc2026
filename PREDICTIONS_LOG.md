# Predictions log (locked before kickoff)

Model: form + Elo features. 3-way via multinomial softmax. Neutral ground.

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

| | logreg | XGBoost | market | result |
|---|---|---|---|---|
| Netherlands win | 35.7% | 35.3% | (tbd) | (tbd) |
| Draw | 30.2% | 28.6% | (tbd) | (tbd) |
| Japan win | 34.1% | 36.1% | (tbd) | (tbd) |

Model read: a true 3-way coin flip, high draw chance. Japan likely underrated
by the market (same pattern as Morocco). To be checked after the match.

> Reproduce: `python predict_match.py "Netherlands" "Japan"`
