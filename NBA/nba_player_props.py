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

THRESHOLDS = {
    "Points": [10, 15, 20, 25, 30, 35, 40, 45, 50],
    "Rebounds": [4, 6, 8, 10, 12, 14, 16],
    "Assists": [4, 6, 8, 10, 12, 14, 16],
    "Threes": [1, 2, 3, 4, 5, 6, 7],
    "Steals": [1, 2, 3, 4, 5],
    "Blocks": [1, 2, 3, 4, 5],
}

MARKETS_FULL = [
    "player_points_alternate", "player_points",
    "player_rebounds_alternate", "player_rebounds",
    "player_assists_alternate", "player_assists",
    "player_threes_made_alternate", "player_threes_made", "player_3pt_made",
    "player_steals_alternate", "player_steals",
    "player_blocks_alternate", "player_blocks",
]

# Fallback: markets likely to be posted first
MARKETS_FALLBACK = [
    "player_points", "player_threes_made"
]

EXCLUDED_BOOKS = {"BetOnline.ag", "Bally Bet", "Fliff", "ReBet"}

MARKET_TO_GROUP = {
    "player_points_alternate": "Points",
    "player_points": "Points",
    "player_rebounds_alternate": "Rebounds",
    "player_rebounds": "Rebounds",
    "player_assists_alternate": "Assists",
    "player_assists": "Assists",
    "player_threes_made_alternate": "Threes",
    "player_threes_made": "Threes",
    "player_3pt_made": "Threes",
    "player_steals_alternate": "Steals",
    "player_steals": "Steals",
    "player_blocks_alternate": "Blocks",
    "player_blocks": "Blocks",
}

def american_to_decimal_one(x):
    try:
        x = float(x)
    except Exception:
        return None
    return (x / 100.0) + 1.0 if x > 0 else (100.0 / abs(x)) + 1.0

def nb_window_now_to_tomorrow():
    tz_et = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(tz_et)
    end_et = (now_et + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    end_utc = end_et.astimezone(timezone.utc)
    return now_utc, end_utc

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

def fetch_bulk_odds(markets_list):
    if not API_KEY:
        print("ERROR: THE_ODDS_API_KEY is not set in your environment.")
        sys.exit(1)

    url = f"{API_BASE}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us,us2",
        "markets": ",".join(markets_list),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json() or []

def build_rows(events_json):
    start_utc, end_utc = nb_window_now_to_tomorrow()
    rows = []
    kept_events = 0

    for ev in events_json:
        try:
            dt_utc = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        except Exception:
            continue
        if not (start_utc <= dt_utc <= end_utc):
            continue
        kept_events += 1

        home, away = ev.get("home_team"), ev.get("away_team")
        event_label = f"{away} @ {home}"

        for book in ev.get("bookmakers", []) or []:
            title = book.get("title") or book.get("key") or "Book"
            if title in EXCLUDED_BOOKS:
                continue
            book_name = (book.get("title") or book.get("key") or "Book").replace(" ", "")
            for market in book.get("markets", []) or []:
                mkey = market.get("key")
                group = MARKET_TO_GROUP.get(mkey)
                if not group:
                    continue
                for outcome in market.get("outcomes", []) or []:
                    price = outcome.get("price")
                    if price is None:
                        continue
                    player = outcome.get("description") or outcome.get("name") or "Unknown"
                    thr = normalize_threshold(group, outcome.get("point"))
                    if not thr:
                        continue
                    rows.append({
                        "event": event_label,
                        "player": player,
                        "group": group,
                        "threshold": thr,
                        "book": book_name,
                        "odds": price,
                    })

    return rows

def pivot_and_score(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index=["event", "player", "group", "threshold"],
        columns="book",
        values="odds",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    if pivot.empty:
        return pd.DataFrame()

    def compute_values(subdf: pd.DataFrame) -> pd.Series:
        keep_cols = [c for c in subdf.columns if c not in ["event", "player", "group", "threshold"]]
        odds_only = subdf[keep_cols]
        decs = odds_only.map(american_to_decimal_one)
        row_decs = decs.iloc[0].dropna()
        if row_decs.empty:
            return pd.Series({
                "best_book": None,
                "best_odds": None,
                "best_decimal": None,
                "avg_other": None,
                "value_ratio": None,
            })
        best_col = row_decs.idxmax()
        best_val = float(row_decs.max())
        avg_other = float(row_decs.drop(labels=[best_col]).mean()) if row_decs.size > 1 else None
        ratio = (best_val / avg_other) if avg_other else None
        best_odds = odds_only.iloc[0][best_col]

        return pd.Series({
            "best_book": best_col,
            "best_odds": best_odds,
            "best_decimal": best_val,
            "avg_other": avg_other,
            "value_ratio": ratio,
        })

    metrics = pivot.groupby(["event", "player", "group", "threshold"], as_index=False).apply(compute_values).reset_index(drop=True)
    out = pd.merge(pivot, metrics, on=["event", "player", "group", "threshold"], how="left")
    if "value_ratio" in out.columns:
        out = out.sort_values(by=["value_ratio"], ascending=[False], na_position="last")
    return out

if __name__ == "__main__":
    # Pass 1: full market set
    try:
        bulk = fetch_bulk_odds(MARKETS_FULL)
    except requests.HTTPError as e:
        print(f"[WARN] bulk fetch (full) HTTP error: {e}")
        bulk = []

    rows = build_rows(bulk)
    if not rows:
        print("[INFO] No rows from full market set. Retrying with fallback markets (points, threes)...")
        time.sleep(0.5)
        try:
            bulk2 = fetch_bulk_odds(MARKETS_FALLBACK)
        except requests.HTTPError as e:
            print(f"[WARN] bulk fetch (fallback) HTTP error: {e}")
            bulk2 = []
        rows = build_rows(bulk2)

    out = pivot_and_score(rows)
    if out.empty:
        print("No NBA data fetched after processing odds (window ok, but no matching markets/thresholds).")
        sys.exit(0)

    out.to_csv("nba_player_props.csv", index=False)
    print("Saved nba_player_props.csv with", len(out), "rows.")
