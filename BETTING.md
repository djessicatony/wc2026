# Betting backtest — does the model's edge make money?

A model that predicts well isn't the same as a model that makes money. This is
the honest test: take the model's probabilities, compare them to the **closing
Polymarket odds**, size bets with the **Kelly criterion**, and simulate a
bankroll. Paper only — no real bets.

- `collect_polymarket.py` — pulls each WC match's closing Polymarket prices
  (last trade before kickoff) for home / draw / away. 48/48 matches collected
  (the full group stage).
- `predict_wc_backtest.py` — predicts every match **leakage-safe** (model trained
  only on pre-WC data; features use only earlier matches), then runs Kelly.
- `kelly_sim.py` — a smaller hand-checked version on the first few matches.

## Result across the group stage (out-of-sample)

| strategy | 20 matches | 32 matches | **48 matches (full group)** |
|---|---|---|---|
| naive Kelly | −42% | −0.3% | **−10.6%** |
| capped 5% | −13% | +45% | **+40.3%** |
| draws-only | +24% | +73% | **+76.8%** |

Model winner accuracy **62.5%** vs the market's **64.6%**.

Watch the naive-Kelly column move −42% → −0.3% → −10.6% as the sample grows —
that wandering *is* the lesson: at this sample size the number hasn't converged
to anything.

## What this actually says

1. **The market is sharper.** The model picks winners less often than just
   backing the betting favourite (62.5% vs 64.6%). On raw prediction we don't beat it.

2. **The ROI numbers are dominated by variance.** Naive Kelly read −0.3% at 32
   matches and −10.6% at 48 — adding 16 matches *moved it the wrong way*, and a
   handful of draws/upsets landing at ~3.5× odds swing the bankroll by tens of
   percent. **At ~50 bets, betting ROI is still noise** — you cannot conclude a
   strategy is profitable from it. (Same small-sample lesson as the rest of this
   repo, now in money.)

3. **Naive value-betting loses or breaks even.** Betting the model's raw
   disagreements with a sharp market mostly bleeds — the big "edges" are the model
   over-rating underdogs, not real value.

4. **The draws-only edge is the one live hypothesis.** It's positive in all three
   samples (+24% → +73% → +76.8%) and has a known mechanism (casual money avoids
   draws, so draw odds are often too long). But beware: this group stage drew
   **29.2%** of the time (well above the ~22% norm), and the model's biggest "draw
   edges" were lopsided matches it gave ~22% that the market priced at 7–13% —
   several of which (Spain–Cape Verde 0–0, England–Ghana 0–0, Ecuador–Curaçao 0–0)
   landed at long odds. That is exactly what a draw-heavy fluke looks like. Even
   this needs **100+ matches** across normal and dry tournaments to confirm, not 48.

## How to actually test it (the only honest path)

Pre-register the hypothesis (draws are underpriced; bet draws where the model
slightly exceeds the market) and grow the out-of-sample set over time:

```bash
python collect_polymarket.py     # after each matchday, refresh closing odds
python predict_wc_backtest.py    # re-run the backtest on the bigger sample
```

If draws-only stays positive across 100–200 matches, the edge is real. If it
reverts to zero, the +76.8% was variance riding a draw-heavy group stage. **That
test — not the 48-match number — is the real result.**
