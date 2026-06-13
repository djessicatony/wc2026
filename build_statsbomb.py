"""v2: pull StatsBomb events and aggregate into rich per-match features.

For each match we download all events (~2.8MB) and compute, per team:
xG, shots, accuracy, possession (via passes), goals. Raw events are
discarded — we keep only aggregates, otherwise it's gigabytes in memory.
"""

import json
import sys
import urllib.request
import pandas as pd

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# international men's tournaments only (where national teams play)
COMPS = [(43, 3), (43, 106), (43, 51), (43, 54), (43, 55),  # World Cups, various years
         (223, 282), (1267, 107), (55, 43), (55, 282)]       # Copa, AFCON, Euro


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def aggregate_match(match_id, home, away, hs, aw, date):
    """match events -> 2 rows (one per team) with rich features"""
    ev = fetch(f"{RAW}/events/{match_id}.json")
    teams = {home: {}, away: {}}
    for t in (home, away):
        shots = [e for e in ev if e.get("type", {}).get("name") == "Shot"
                 and e.get("team", {}).get("name") == t]
        passes = [e for e in ev if e.get("type", {}).get("name") == "Pass"
                  and e.get("team", {}).get("name") == t]
        on_target = [s for s in shots
                     if s["shot"]["outcome"]["name"] in ("Goal", "Saved", "Saved To Post")]
        teams[t] = {
            "xg": sum(s["shot"]["statsbomb_xg"] for s in shots),
            "shots": len(shots),
            "on_target": len(on_target),
            "passes": len(passes),
        }
    total_passes = teams[home]["passes"] + teams[away]["passes"] or 1
    rows = []
    for t, opp, gf, ga in [(home, away, hs, aw), (away, home, aw, hs)]:
        s = teams[t]
        rows.append({
            "match_id": match_id, "date": date, "team": t, "opponent": opp,
            "goals_for": gf, "goals_against": ga,
            "xg_for": s["xg"], "xg_against": teams[opp]["xg"],
            "shots": s["shots"],
            "shot_accuracy": s["on_target"] / s["shots"] if s["shots"] else 0,
            "possession": s["passes"] / total_passes,
            "won": int(gf > ga),
        })
    return rows


all_rows = []
seen = set()
for comp_id, season_id in COMPS:
    try:
        matches = fetch(f"{RAW}/matches/{comp_id}/{season_id}.json")
    except Exception as e:
        print(f"  skip {comp_id}/{season_id}: {e}", file=sys.stderr)
        continue
    for m in matches:
        mid = m["match_id"]
        if mid in seen:
            continue
        seen.add(mid)
        try:
            all_rows += aggregate_match(
                mid, m["home_team"]["home_team_name"], m["away_team"]["away_team_name"],
                m["home_score"], m["away_score"], m["match_date"])
        except Exception as e:
            print(f"  match {mid} skipped: {e}", file=sys.stderr)
        if len(seen) % 25 == 0:
            print(f"matches processed: {len(seen)}, rows: {len(all_rows)}", flush=True)

df = pd.DataFrame(all_rows)
df.to_csv("data/statsbomb_team_matches.csv", index=False)
print(f"\ndone: {len(df)} team-perspective rows from {len(seen)} matches -> data/statsbomb_team_matches.csv")
