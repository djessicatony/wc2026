# 🥊 MMA predictor (UFC White House, June 15 2026)

The football playbook applied to UFC — fighter Elo + composite skill ratings
instead of team form. Benchmarked against Polymarket and
**[leo.taps](https://www.instagram.com/leo.taps/)** (an MMA analyst with a similar model).

Honest headline: **in MMA the market is hard to beat.** Backtest ~65% vs the
market's ~70% — the opposite of football. One punch ends a fight, and the market
prices in style and intangibles that raw stats miss.

<p align="center">
  <img src="assets/mma_predictions.png" width="80%">
</p>

| Fight | Our model | leo.taps | Polymarket | Result |
|---|---|---|---|---|
| Topuria – Gaethje | Topuria 78% | Topuria 91% | Topuria 80% | **Gaethje** TKO R4 ✗ all |
| Pereira – Gane | **Gane 56%** | Pereira 62% | ~coin flip | **Gane** TKO R2 — **only our model** ✓ |
| Ruffy – Chandler | Ruffy 67% | Ruffy 90% | Ruffy 81% | **Ruffy** TKO R1 ✓ |
| O'Malley – Zahabi | O'Malley 61% | — | O'Malley 80% | **O'Malley** TKO R2 ✓ |

**Scoreboard: our model 3/4, Polymarket 2/4, leo.taps 1/3.** The model beat the
market on the night — driven by one contrarian call: it was the *only* one of the
three to pick **Gane** over Pereira, and Gane won by 2nd-round TKO. Topuria's loss
was a genuine upset everyone missed (market 80%, leo.taps 91%) — an 80% favourite
still loses 1 in 5 times. (One card is a tiny sample; the backtest still says the
market is sharper long-run.)

**Specific-outcome call (for fun):** combining winner + method, the model's top pick
for the main event was **Topuria by KO, rounds 1–2** (~33%). It missed (Gaethje won
by R4 TKO), but the *type* — a finish — was right; every fight on the card was a TKO.
See `finger_in_air.py`.

<p align="center">
  <img src="assets/mma_reliability.png" width="42%">
  <img src="assets/mma_sensitivity_topuria.png" width="55%">
</p>

Predictions were stable across Elo settings and training windows (Topuria 78–79%,
Gane 55–57%).

**What this build mostly taught: debugging and skepticism.**
- An early run predicted the underdog at 64% — a bug, not an edge: mixed
  difference-sign conventions plus a "Red corner wins 58%" bias leaking in. Fixed by
  computing all features one way and averaging over both corners.
- Tried to turn an analyst's "Gane is southpaw, open stance" read into a feature —
  but the data (and UFC.com) say **both fighters are orthodox**. The premise was
  wrong, which flips the argument. Verify the input before modelling it.
- Finish method and round are predictable in principle but barely beat guessing —
  the finer the question, the more it's just randomness.

Scripts: `train_mma.py`, `predict_mma.py`, `predict_composite.py`, `predict_stance.py`,
`predict_method.py`, `predict_method_round.py`, `finger_in_air.py`, `make_viz_mma.py`.
