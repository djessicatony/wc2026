# Betting backtest — does the model's edge make money?

A model that predicts well isn't the same as a model that makes money. This is
the honest test: take the model's probabilities, compare them to the **closing
Polymarket odds**, size bets with the **Kelly criterion**, and simulate a
bankroll. Paper only — no real bets.

- `collect_polymarket.py` — pulls each WC match's closing Polymarket prices
  (last trade before kickoff) for home / draw / away. 32/32 matches collected.
- `predict_wc_backtest.py` — predicts every match **leakage-safe** (model trained
  only on pre-WC data; features use only earlier matches), then runs Kelly.
- `kelly_sim.py` — a smaller hand-checked version on the first few matches.

## Result on 32 World Cup matches (out-of-sample)

| strategy | 20 matches | 32 matches |
|---|---|---|
| naive Kelly | −42% | −0.3% |
| capped 5% | −13% | +45% |
| draws-only | +24% | +73% |

Model winner accuracy **56%** vs the market's **59%**.

## What this actually says

1. **The market is sharper.** The model picks winners less often than just
   backing the betting favourite (56% vs 59%). On raw prediction we don't beat it.

2. **The ROI numbers are dominated by variance.** Adding 12 matches swung capped
   Kelly from −13% to +45%. A handful of draws/upsets landing at ~3.5× odds move
   the bankroll by tens of percent. **At ~30 bets, betting ROI is noise** — you
   cannot conclude a strategy is profitable from it. (Same small-sample lesson as
   the rest of this repo, now in money.)

3. **Naive value-betting loses or breaks even.** Betting the model's raw
   disagreements with a sharp market mostly bleeds — the big "edges" are the model
   over-rating underdogs, not real value.

4. **The draws-only edge is the one live hypothesis.** It's positive in both
   samples and has a known mechanism (casual money avoids draws, so draw odds are
   often too long). But the wild swings mean even this needs **100+ matches** to
   confirm, not 32.

## How to actually test it (the only honest path)

Pre-register the hypothesis (draws are underpriced; bet draws where the model
slightly exceeds the market) and grow the out-of-sample set over time:

```bash
python collect_polymarket.py     # after each matchday, refresh closing odds
python predict_wc_backtest.py    # re-run the backtest on the bigger sample
```

If draws-only stays positive across 100–200 matches, the edge is real. If it
reverts to zero, the +73% was variance. **That test — not the 32-match number —
is the real result.**
