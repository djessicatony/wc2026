"""Paper Kelly betting simulator (no real money).

For each outcome: if our model's probability > the market price (= implied
probability), we have an edge. Bet a fraction of the bankroll using
half-Kelly:  f = 0.5 * (p - price) / (1 - price).
Payout on a win = stake * (1/price - 1); on a loss = -stake.

Tests the real question: does the model's "edge" actually make money?
"""

START = 100.0
KELLY = 0.5  # half-Kelly (less variance)

# each market: list of (outcome, model_prob, market_price), and the winner
FOOTBALL = [
    ("Brazil–Morocco", [("Brazil", .36, .59), ("Draw", .28, .26), ("Morocco", .36, .17)], "Draw"),
    ("Netherlands–Japan", [("Netherlands", .36, .48), ("Draw", .30, .28), ("Japan", .34, .26)], "Draw"),
    ("France–Senegal", [("France", .59, .67), ("Draw", .25, .22), ("Senegal", .16, .13)], "France"),
    ("England–Croatia", [("England", .46, .59), ("Draw", .30, .25), ("Croatia", .24, .17)], "England"),
]
MMA = [
    ("Topuria–Gaethje", [("Topuria", .78, .80), ("Gaethje", .22, .20)], "Gaethje"),
    ("Pereira–Gane", [("Pereira", .44, .51), ("Gane", .56, .49)], "Gane"),
    ("Ruffy–Chandler", [("Ruffy", .67, .81), ("Chandler", .33, .19)], "Ruffy"),
    ("O'Malley–Zahabi", [("O'Malley", .61, .80), ("Zahabi", .39, .20)], "O'Malley"),
]


def simulate(card, name, edge_cap=None):
    """edge_cap: trust the sharp market — treat any model edge above this as
    just edge_cap (so huge deviations don't get huge bets)."""
    bank = START
    tag = f"cap {edge_cap:.0%}" if edge_cap else "naive"
    print(f"\n===== {name} ({tag}, start ${START:.0f}) =====")
    for match, outcomes, winner in card:
        pnl = 0.0
        for outcome, p, price in outcomes:
            edge = p - price
            if edge <= 0:
                continue
            if edge_cap:
                edge = min(edge, edge_cap)
            stake = KELLY * edge / (1 - price) * bank
            pnl += stake * (1 / price - 1) if outcome == winner else -stake
        bank += pnl
    roi = (bank - START) / START * 100
    print(f"  FINAL: ${bank:.2f}  (ROI {roi:+.1f}%)")
    return bank


def draws_only(card):
    """domain bet: only back the DRAW when the model slightly exceeds the market"""
    bank = START
    for match, outcomes, winner in card:
        for outcome, p, price in outcomes:
            if outcome != "Draw" or p <= price:
                continue
            stake = KELLY * (p - price) / (1 - price) * bank
            bank += stake * (1 / price - 1) if winner == "Draw" else -stake
    print(f"  DRAWS-ONLY: ${bank:.2f}  (ROI {(bank-START)/START*100:+.1f}%)")


print("FOOTBALL:")
simulate(FOOTBALL, "FOOTBALL", edge_cap=None)
simulate(FOOTBALL, "FOOTBALL", edge_cap=0.05)
draws_only(FOOTBALL)
print("\nMMA:")
simulate(MMA, "MMA", edge_cap=None)
simulate(MMA, "MMA", edge_cap=0.05)
