# Post-mortem: Brazil 1–1 Morocco (World Cup 2026, June 14)

Predictions were locked **before kickoff**. Here's how they held up.

## Prediction vs reality

| | Our model (v5, 3-way) | Polymarket | Reality |
|---|---|---|---|
| Brazil win | 36% | **59%** | — |
| Draw | 28% | 26% | **✓ 1–1** |
| Morocco win | 36% | 17% | — |

Binary "does Brazil win?": our v1 said 29%, v3 (with Elo) said 37%, market said 59%.
**Brazil did not win.** The data-driven model leaned the right way; the market overrated Brazil.

## The killer stat: expected goals (xG)

```
xG:  Morocco 1.24   —   Brazil 1.28
```

Essentially **dead even** — exactly the coin-flip our model described (36/36), and the
opposite of the market's 59/17. Morocco scored first (Saibari), outplayed Brazil for
long stretches, and Brazil was rescued only by Vinícius Jr.'s individual quality.
Possession Brazil 54%; Brazil more clinical (33% shots on target vs 17%).

## Why our model was right and the market wasn't

The model reads **results** (form + Elo). Elo had it Brazil 1948 vs Morocco 1906 —
near-equal, because Morocco is genuinely strong (2022 WC semifinalists). The xG tie
confirmed the match was balanced. The market overweighted Brazil's **brand and squad
quality** — real factors our features don't capture, but ones the hype overrated here.

## Honest caveat

One match doesn't prove the model beats the market in general — both are judged over
many predictions. And the draw (28%) was actually below our two "win" probabilities,
so the model didn't "predict a draw" — it predicted an **even contest**, and an even
contest is what happened (the xG tie is the real vindication, not the scoreline).

## What this confirms

- Our **Elo feature** saw Morocco's strength the market underrated → data > hype here.
- The model's probabilities are **calibrated** (see v8), so "even match" was an honest read.
- Next feature frontier to close the gap to the market: **squad/player quality**
  (the Vinícius factor) — asymmetric, would carry real signal.
