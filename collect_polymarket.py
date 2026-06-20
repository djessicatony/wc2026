"""Collect closing Polymarket odds for every played World Cup 2026 match.

For each match: find the Polymarket event, read its 3 markets (home win /
draw / away win), and take each market's last traded price BEFORE kickoff
(gameStartTime) = the closing implied probability. Saves data/wc_polymarket.csv.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
import pandas as pd

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# CSV name -> the name Polymarket uses
ALIAS = {"Turkey": "Türkiye", "Czech Republic": "Czechia", "Ivory Coast": "Côte d'Ivoire",
         "Cape Verde": "Cabo Verde", "Bosnia and Herzegovina": "Bosnia-Herzegovina",
         "South Korea": "Korea Republic"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def norm(s):
    return str(s).lower().replace(".", "").replace("-", " ").strip()


def pm(team):
    return ALIAS.get(team, team)


def closing_price(token_id, kickoff_ts):
    try:
        d = get(f"{CLOB}/prices-history?market={token_id}&interval=max&fidelity=60")
    except Exception:
        return None
    pre = [p for p in d.get("history", []) if p["t"] <= kickoff_ts]
    return round(pre[-1]["p"], 3) if pre else None


def find_event(home, away, match_date):
    q = urllib.parse.quote(f"{pm(home)} {pm(away)}")
    try:
        res = get(f"{GAMMA}/public-search?q={q}&limit_per_type=10")
    except Exception:
        return None
    for e in res.get("events", []):
        slug = e.get("slug", "")
        t = norm(e.get("title", ""))
        if slug.startswith("fif") and norm(pm(home)) in t and norm(pm(away)) in t:
            return e["id"]
    return None


raw = pd.read_csv("data/international_results.csv", parse_dates=["date"])
wc = raw[(raw.tournament == "FIFA World Cup") & (raw.date >= "2026-06-01")].dropna(subset=["home_score"])

rows, miss = [], []
for _, m in wc.iterrows():
    h, a = m.home_team, m.away_team
    eid = find_event(h, a, m.date)
    if not eid:
        miss.append(f"{h} vs {a}"); continue
    ev = get(f"{GAMMA}/events/{eid}")
    kt = ev.get("startTime") or ev.get("gameStartTime") or ev.get("endDate")
    kickoff = int(datetime.fromisoformat(kt.replace("Z", "+00:00")).timestamp())
    prices = {"home": None, "draw": None, "away": None}
    for mk in ev.get("markets", []):
        qn = norm(mk.get("question", ""))
        toks = json.loads(mk["clobTokenIds"])
        side = "draw" if "draw" in qn else "home" if norm(pm(h)) in qn else "away" if norm(pm(a)) in qn else None
        if side:
            prices[side] = closing_price(toks[0], kickoff)
    # one illiquid side with no pre-kickoff trades → infer from the others (probs ~sum to 1)
    missing = [k for k, v in prices.items() if v is None]
    if len(missing) == 1:
        prices[missing[0]] = round(max(0.01, 1 - sum(v for v in prices.values() if v is not None)), 3)
    if all(v is not None for v in prices.values()):
        res = "home" if m.home_score > m.away_score else "away" if m.home_score < m.away_score else "draw"
        rows.append({"date": m.date.date(), "home": h, "away": a,
                     "mkt_home": prices["home"], "mkt_draw": prices["draw"], "mkt_away": prices["away"],
                     "result": res, "score": f"{int(m.home_score)}-{int(m.away_score)}"})
        print(f"  OK  {h} vs {a}: H{prices['home']} D{prices['draw']} A{prices['away']} → {res}")
    else:
        miss.append(f"{h} vs {a} (partial {prices})")
    time.sleep(0.3)

df = pd.DataFrame(rows)
df.to_csv("data/wc_polymarket.csv", index=False)
print(f"\ncollected {len(df)}/{len(wc)} matches → data/wc_polymarket.csv")
for x in miss:
    print("  missed:", x)
