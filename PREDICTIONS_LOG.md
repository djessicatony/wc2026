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
| France win | 58.8% | 67% | **✓ 3–1 France** |
| Draw | 24.8% | 22% | — |
| Senegal win | 16.4% | 13% | — |

Result: France won 3–1. Not a contrarian call — both the model and the market had
France as favourite, and France delivered. Both right; the model was just a touch
less confident (59% vs 67%).

## England vs Croatia — World Cup 2026, June 18

Elo: England 1944, Croatia 1886 (58-point gap — close).

| | our logreg | market | result |
|---|---|---|---|
| England win | 46.3% | 59% | **✓ 4–2 England** |
| Draw | 30.0% | 25% | — |
| Croatia win | 23.7% | 17% | — |

Result: England won 4–2. The contrarian "near-even" read **missed** — England were
clear, and the market (59%) was right. The "more even than the market" pattern is
now 2 of 3 (Brazil ✓, Netherlands ✓, England ✗). Honest reminder: it's a tendency,
not a law, and a small sample.

Here the two data models **split**: ours reads an even match, while sujar.tech
(53% Netherlands) sides with the market. We are the contrarian — most bullish
on Japan (34% vs market 26% vs sujar.tech 18%). To be checked after the match.

> Reproduce: `python predict_match.py "Netherlands" "Japan"`

---

## Group-stage backtest — all 48 played matches (June 11–23)

Once the whole group stage was played, I stopped cherry-picking 4 matches and
graded the model **out-of-sample on every played match** (`wc_accuracy.py`,
trained only on pre-June-11 games, leakage-safe Elo + form).

| metric | value | note |
|---|---|---|
| winner accuracy | **62.5%** (30/48) | exactly ties the "pick higher Elo" baseline |
| log loss | **0.893** | beats a 3-way coin flip (1.099) — probs are informative |
| actual draw rate | **29.2%** (14/48) | a draw-heavy group stage |
| model mean P(draw) | 22.6% | model under-expected draws by ~7 pts |
| times model picked draw | **0 / 48** | the draw is never the single most likely outcome |

**The headline is the draw problem.** 12 of the model's 18 misses were draws it
called as home wins. On the 14 matches that actually drew, the model's average
P(draw) was 21.3% — *lower* than the 23.2% it gave the non-draws. So it had no
signal at all for *which* matches would draw; softmax logreg almost never makes
"draw" the argmax because a draw is rarely the most likely single result even
when it's underpriced. Winner accuracy tying the trivial Elo baseline says the
form features add nothing here — Elo alone carries the model. The probabilities
are still useful (log loss < coin flip), which is why the betting edge lives in
draw *prices*, not in picking winners (see `BETTING.md`).

> Reproduce: `python wc_accuracy.py`
