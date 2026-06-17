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

## France vs Senegal — World Cup 2026, June 17

Elo: France 1988, Senegal 1815 (173-point gap). France's form clearly better.

| | our logreg | market | result |
|---|---|---|---|
| France win | 58.8% | 67% | (tbd) |
| Draw | 24.8% | 22% | (tbd) |
| Senegal win | 16.4% | 13% | (tbd) |

Not a contrarian call this time — the data agrees France is the favourite, just a
bit less confident than the market (59% vs 67%). No Senegal upset shout from the
model; the Elo gap and form favour France clearly.

## England vs Croatia — World Cup 2026, June 18

Elo: England 1944, Croatia 1886 (58-point gap — close).

| | our logreg | market | result |
|---|---|---|---|
| England win | 46.3% | 59% | (tbd) |
| Draw | 30.0% | 25% | (tbd) |
| Croatia win | 23.7% | 17% | (tbd) |

Contrarian again: the market backs England (59%), the model has it near-even —
England below 50%, with the draw + Croatia outweighing. Third football match where
the model reads it more even than the market and rates the underdog higher.

Here the two data models **split**: ours reads an even match, while sujar.tech
(53% Netherlands) sides with the market. We are the contrarian — most bullish
on Japan (34% vs market 26% vs sujar.tech 18%). To be checked after the match.

> Reproduce: `python predict_match.py "Netherlands" "Japan"`
