# Does the starting lineup close the gap to the market?

`POSTMORTEM.md` ended by naming the model's next frontier: **squad/player quality**
— the "Vinícius factor" the Elo feature can't see. This is the investigation of
whether that signal is worth building, and the honest verdict.

## The idea

Elo rates a *team*, but it can't see **who actually plays**. When a side that has
already qualified rests its A-team in a dead-rubber group finale, Elo still rates
them at full strength — but they aren't. Confirmed XIs land at a scrapeable moment
(~2h before kickoff, e.g. Fabrizio Romano). So the plan was: player ratings
(Sofascore/FBref) → aggregate the **actual** XI → team strength that reacts to
rotation → feed the model. Optionally a Twitter/X cron on the lineup source.

## Don't build the pipeline to find out whether it works

Before scraping anything, a 30-minute manual smell test: take the two group
finales where an A-team rested, **lower the resting team's Elo** by a few deltas,
and see whether the prediction moves *toward* reality and the market. Two matches
is anecdote, not proof — just a directional check (`lineup_test.py`).

### Match 1 — Norway vs France (Norway rested Haaland & co.) → **1–4 France**

| | Norway | draw | France |
|---|---|---|---|
| base model | 21% | 25% | 55% |
| − 80 Elo Norway | 13% | 21% | **66%** |
| − 150 Elo Norway | 8% | 18% | 74% |

France dominated 4–1, so down-weighting Norway pushed the call toward reality.
**The signal helped.** And the payoff line: **Polymarket had France ~62%** — sitting
exactly between the base (55%) and the −80 down-weight (66%). The gap between our
model and the market *was* the lineup. The POSTMORTEM hypothesis, confirmed with a
number.

### Match 2 — Jordan vs Argentina (Argentina rested Messi) → **1–3 Argentina**

| | Jordan | draw | Argentina |
|---|---|---|---|
| base model | 3% | 13% | 84% |
| − 120 Elo Argentina | 7% | 19% | **74%** |

Argentina won comfortably 3–1 **without Messi**. Down-weighting them pulled the
prediction *away* from reality. **The signal hurt.**

## Verdict: real, but not worth the schlep

The naive rule "star rested → down-weight" went **1-for-2**. The difference is
mechanical:
- **Norway *is* Haaland**, and the match was competitive → resting him tips it.
- **Argentina without Messi is still miles deeper than Jordan** → resting him is noise.

So the signal is **conditional**: it only matters when (a) the rested player is a
large share of team strength *and* (b) the match is close. In blowouts it's noise.
Three facts kill the heavy build:

1. **Conditional** — a naive "subtract missing players" feature mis-fires (the Argentina case).
2. **Rare** — A-team rotation happens mostly in already-decided group finales: ~5 of ~100 matches. Big effect on those, tiny effect on overall accuracy (classic diminishing returns).
3. **Already priced** — the market moves on the confirmed XI within minutes, so this can't beat it (it *explains* the model↔market gap; it doesn't beat the market).

**Building Sofascore scraping + a Twitter cron + name-matching for a marginal, rare,
already-priced feature is not worth it.** The cheap manual test already delivered
the real result: on competitive matches, the model-vs-market gap is the lineup.
That insight is the deliverable — the infrastructure isn't needed.

If ever pursued, the *smart* version (not the naive one): effective-XI strength
weighted by squad depth, applied only to near-50/50 matches — not "sum of ratings
minus absentees."

> Reproduce: `python lineup_test.py`
> Related: `POSTMORTEM.md` (raised the hypothesis), `BETTING.md` (market is sharp).
