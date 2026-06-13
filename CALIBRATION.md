# Calibration check: our model vs Polymarket

The best sanity check for a probability is a real-money prediction market.
Polymarket on this match had **$12.42M volume** — a strong gold standard.

## The gap

| Outcome | Our model | Polymarket |
|---|---|---|
| Brazil win | **29.1%** | **59%** |
| Draw | (folded into "not win") | 26% |
| Morocco win | (folded into "not win") | 17% |
| Brazil NOT win | 70.9% | 43% |

Our model **badly underrated Brazil** — half the market's number. A single model
disagreeing this much with a deep market is almost always the model's fault, not
the market's. So this is a calibration failure worth diagnosing.

## Diagnosis: no "strength of schedule"

The model's features (win rate, goals for/against) treat every match equally.
But the last 10 opponents were not equal:

- **Morocco** padded stats vs weak teams: Madagascar 4:0, Burundi 5:0, Tanzania,
  Zambia. Their stellar 0.30 goals-conceded is largely against minnows.
- **Brazil** faced top sides: France (lost 1:2), Croatia, Japan, Senegal. Their
  "worse" numbers come from much harder opposition.

The model counts a clean sheet vs Madagascar the same as one vs France. It has
**no notion of opponent quality**, so it overrates Morocco's easy run and
underrates Brazil's tough one. The market prices in team strength; we don't.

## The fix (v3)

Add opponent-strength features — exactly the "win rate vs dominant teams" idea
from day one:
- opponent's FIFA ranking / Elo rating at match time
- form weighted by opponent strength
- goals adjusted for opponent quality

This is the single highest-value missing feature, and the market gap proves it.

## The lesson

A model can be internally honest (65% backtest, no leakage, no overfitting) and
still be **miscalibrated on the real question** because a key signal is missing
from the features. Comparing against a market (or any strong external baseline)
is how you catch it. Backtest accuracy tells you the model is consistent;
the market tells you whether it's *right*.
