import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

API_KEY = os.getenv("THE_ODDS_API_KEY", "")
API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"
REGIONS = "us,us2"

EXCLUDED_BOOKS = {
    "BetOnline.ag", "Bally Bet", "Fliff", "ReBet",
    "SuperBook", "Tipico", "Betway", "SI Sportsbook"
}

MARKETS = [
    "player_points_alternate",
    "player_rebounds_alternate",
    "player_assists_alternate",
    "player_threes_made_alternate",
    "player_steals_alternate",
    "player_blocks_alternate",
]

def et_window_today_tomorrow():
    tz_et = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(tz_et)
    end_et = (now_et + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    return now_utc, end_et.astimezone(timezone.utc)

def fetch_events():
    url = f"{API_BASE}/sports/{SPORT_KEY}/events"
    params = {"apiKey": API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json() or []

def fetch_event_odds(event_id: str):
    url = f"{API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": ",".join(MARKETS),
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=30)
    return r

def main():
    if not API_KEY:
        print("ERROR: THE_ODDS_API_KEY not set in environment.")
        return 1

    start_utc, end_utc = et_window_today_tomorrow()
    events = fetch_events()
    kept = [
        e for e in events
        if start_utc <= datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= end_utc
    ]

    print(f"Found {len(kept)} NBA events in ET window.\n")
    if not kept:
        return 0

    for e in kept:
        ev_id = e["id"]
        label = f"{e.get('away_team')} @ {e.get('home_team')}"
        print(f"===== {label} =====")
        r = fetch_event_odds(ev_id)
        print(f"HTTP {r.status_code}")
        if r.status_code not in (200,):
            print("  No data returned or endpoint error.\n")
            continue

        js = r.json()
        if not js or "bookmakers" not in js:
            print("  JSON missing bookmakers key.\n")
            continue

        books = js.get("bookmakers", [])
        print(f"  bookmakers found: {len(books)}")
        for b in books:
            print(f"    - {b.get('title')}")
            mk = [m.get('key') for m in b.get('markets', [])]
            print(f"      markets: {mk}")
            for m in b.get("markets", []):
                outs = len(m.get("outcomes", []) or [])
                print(f"        {m.get('key')}: {outs} outcomes")
        print()

    print("Done debugging.")

if __name__ == "__main__":
    sys.exit(main())
