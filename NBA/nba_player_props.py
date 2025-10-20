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

THRESHOLDS = {
    "Points": [10, 15, 20, 25, 30, 35, 40, 45, 50],
    "Rebounds": [4, 6, 8, 10, 12, 14, 16],
    "Assists": [4, 6, 8, 10, 12, 14, 16],
    "Threes": [1, 2, 3, 4, 5, 6, 7],
    "Steals": [1, 2, 3, 4, 5],
    "Blocks": [1, 2, 3, 4, 5],
}

MARKET_MAP = {
    "player_points_alternate": "Points",
    "player_rebounds_alternate": "Rebounds",
    "player_assists_alternate": "Assists",
    "player_threes_made_alternate": "Threes",
    "player_steals_alternate": "Steals",
    "player_blocks_alternate": "Blocks",
}

MARKETS = list(MARKET_MAP.keys())


def american_to_decimal(x):
    try:
        x = float(x)
    except Exception:
        return None
    return (x / 100.0) + 1.0 if x > 0 else (100.0 / abs(x)) + 1.0


def et_window_today_tomorrow():
    tz_et = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(tz_et)
    end_et = (now_et + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    return now_utc, end_et.astimezone(timezone.utc)


def normalize_threshold(group: str, point: float | None):
    if point is None:
        return None
    try:
        p = float(point)
    except Exception:
        return None
    for t in THRESHOLDS[group]:
        if abs(p - t) <= 0.5:
            return str(t)
    return None


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
    if r.status_code in (404, 422):
        return None
    r.raise_for_status()
    return r.json()


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

    if not kept:
        print("No NBA events found in ET window (today + tomorrow).")
        return 0

    print(f"Found {len(kept)} NBA events in ET window. Fetching props...\n")

    rows = []
    for e in kept:
        ev_id = e["id"]
        label = f"{e.get('away_team')} @ {e.get('home_team')}"
        try:
            js = fetch_event_odds(ev_id)
        except requests.HTTPError as err:
            print(f"  {label} — HTTP {err.response.status_code}")
            continue
        if not js or "bookmakers" not in js:
            print(f"  {label} — no props returned")
            continue

        for book in js.get("bookmakers", []):
            title = book.get("title") or book.get("key") or "Book"
            if title in EXCLUDED_BOOKS:
                continue
            book_name = title.replace(" ", "")
            for m in book.get("markets", []):
                mkey = m.get("key")
                group = MARKET_MAP.get(mkey)
                if not group:
                    continue
                for o in m.get("outcomes", []):
                    player = o.get("description") or o.get("name") or "Unknown"
                    price = o.get("price")
                    thr = normalize_threshold(group, o.get("point"))
                    if not thr or price is None:
                        continue
                    rows.append({
                        "event": label,
                        "player": player,
                        "group": group,
                        "threshold": thr,
                        "book": book_name,
                        "odds": price,
                    })
        print(f"  {label} — {len(rows)} total rows so far")
        time.sleep(0.25)

    if not rows:
        print("No NBA props found for this window.")
        return 0

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index=["event", "player", "group", "threshold"],
        columns="book",
        values="odds",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    def compute_values(row):
        odds = row.drop(labels=["event", "player", "group", "threshold"], errors="ignore")
        decs = odds.map(american_to_decimal)
        decs = decs.dropna()
        if decs.empty:
            return pd.Series({"best_book": None, "best_odds": None, "best_decimal": None, "avg_other": None, "value_ratio": None})
        best_col = decs.idxmax()
        best_val = float(decs.max())
        avg_other = float(decs.drop(labels=[best_col]).mean()) if len(decs) > 1 else None
        ratio = (best_val / avg_other) if avg_other else None
        best_odds = odds.get(best_col)
        return pd.Series({"best_book": best_col, "best_odds": best_odds, "best_decimal": best_val, "avg_other": avg_other, "value_ratio": ratio})

    metrics = pivot.apply(compute_values, axis=1)
    out = pd.concat([pivot, metrics], axis=1)
    out = out.sort_values(by="value_ratio", ascending=False, na_position="last")

    out.to_csv("nba_player_props.csv", index=False)
    print(f"\nSaved nba_player_props.csv with {len(out)} rows.")

if __name__ == "__main__":
    sys.exit(main())
