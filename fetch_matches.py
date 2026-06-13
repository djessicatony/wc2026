"""Fetch team match history from football-data.org into data/*.json.

Boilerplate: an HTTP client with auto-throttling based on response headers.
No feature building here.
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API = "https://api.football-data.org/v4"
TOKEN = os.environ["FOOTBALL_DATA_API_KEY"]

TEAMS = {"brazil": 764, "morocco": 815}
# API caps the window at 750 days, so fetch history in chunks
DATE_CHUNKS = [
    ("2021-06-01", "2023-06-20"),
    ("2023-06-21", "2025-07-09"),
    ("2025-07-10", "2026-06-13"),  # only matches BEFORE ours: no data from the future
]

session = requests.Session()
session.headers["X-Auth-Token"] = TOKEN


def get(path: str, params: dict | None = None) -> dict:
    """GET that respects the rate limit: read headers, wait when needed."""
    while True:
        resp = session.get(f"{API}{path}", params=params, timeout=30)
        remaining = int(resp.headers.get("x-requests-available-minute", "1"))
        reset_sec = int(resp.headers.get("x-requestcounter-reset", "60"))
        if resp.status_code == 429:
            print(f"  rate limited: waiting {reset_sec}s...")
            time.sleep(reset_sec + 1)
            continue
        resp.raise_for_status()
        if remaining == 0:
            print(f"  quota exhausted: pausing {reset_sec}s...")
            time.sleep(reset_sec + 1)
        return resp.json()


def fetch_team(name: str, team_id: int) -> None:
    matches = []
    for date_from, date_to in DATE_CHUNKS:
        data = get(
            f"/teams/{team_id}/matches",
            {"status": "FINISHED", "dateFrom": date_from, "dateTo": date_to, "limit": 500},
        )
        matches.extend(data["matches"])
    out = Path("data") / f"{name}_matches.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(matches, indent=1))
    dates = [m["utcDate"][:10] for m in matches]
    print(f"{name}: {len(matches)} matches, {min(dates)} … {max(dates)} → {out}")


if __name__ == "__main__":
    for name, team_id in TEAMS.items():
        fetch_team(name, team_id)
