import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

################################################################################
# NFL Player Props Odds Scraper (Tail Wing)
# - Targets rushing, receiving, passing thresholds and anytime TD
# - Normalizes alt lines to target thresholds (25/50/80/110 rush/rec, 250/300/350 pass)
# - Compares best price vs average of others (in Decimal odds)
# - Pulls ALL games from NOW (UTC) through END OF MONDAY (US/Eastern). If run on
#   a Monday, window ends tonight (Mon) 23:59:59 ET.
# - Outputs: nfl_player_props.csv
################################################################################

#API_KEY = os.getenv("THE_ODDS_API_KEY")
API_KEY = "7f8cbb98207020adbd0218844a595725"
API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"

# ---- Config: thresholds & markets -------------------------------------------
THRESHOLDS = {
    "Rushing": [25, 50, 80, 110],
    "Receiving": [25, 50, 80, 110],
    "Passing": [250, 300, 350],
}

MARKETS = {
    "Rushing": "player_rush_yds_alternate",
    "Receiving": "player_reception_yds_alternate",
    "Passing": "player_pass_yds_alternate",
    "AnytimeTD": "player_anytime_td",
}

EXCLUDED_BOOKS = {"BetOnline.ag", "Bally Bet", "Fliff", "ReBet"}
# ---- Helpers -----------------------------------------------------------------

def american_to_decimal_one(x):
    """Convert one American odds value to Decimal; return None on failure."""
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return (x / 100.0) + 1.0 if x > 0 else (100.0 / abs(x)) + 1.0


def normalize_threshold(market_key: str, point: float):
    """Map an alt-line 'point' to one of our canonical thresholds with +/-1.0 tol."""
    if market_key == "player_anytime_td":
        return "Anytime"
    group = next((g for g, k in MARKETS.items() if k == market_key), None)
    if not group:
        return None
    if point is None:
        return None
    point = float(point)
    for t in THRESHOLDS[group]:
        if abs(point - t) <= 1.0:
            return str(t)
    return None


def nfl_window_now_to_monday():
    """Return (start_utc, end_utc) from now to end of Monday ET (23:59:59).
    If called on a Monday, it ends tonight.
    """
    tz_et = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(tz_et)

    # Monday=0 .. Sunday=6. Days ahead until next Monday (0 if today is Monday)
    days_until_monday = (7 - now_et.weekday()) % 7
    if days_until_monday == 0:  # today is Monday
        end_et = now_et.replace(hour=23, minute=59, second=59, microsecond=0)
    else:
        next_monday_et = (now_et + timedelta(days=days_until_monday)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )
        end_et = next_monday_et

    start_utc = now_utc
    end_utc = end_et.astimezone(timezone.utc)
    return start_utc, end_utc

# ---- API calls ---------------------------------------------------------------

def fetch_events():
    """Fetch NFL events and filter to [now, end-of-Monday] window in UTC."""
    if not API_KEY:
        print("ERROR: THE_ODDS_API_KEY is not set in your environment.")
        sys.exit(1)

    start_utc, end_utc = nfl_window_now_to_monday()

    url = f"{API_BASE}/sports/{SPORT_KEY}/events"
    params = {"apiKey": API_KEY, "dateFormat": "iso"}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    events = resp.json() or []

    out = []
    for e in events:
        # commence_time is ISO 8601 in UTC with 'Z'
        dt_utc = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if start_utc <= dt_utc <= end_utc:
            out.append(e)
    return out


def fetch_event_odds(event_id: str):
    url = f"{API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us,us2",
        "markets": ",".join(MARKETS.values()),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# ---- Main --------------------------------------------------------------------

if __name__ == "__main__":
    events = fetch_events()
    if not events:
        print("No events found in the now→Monday window.")
        sys.exit(0)

    rows = []

    for ev in events:
        odds_data = fetch_event_odds(ev["id"]) or {}
        home, away = ev.get("home_team"), ev.get("away_team")
        event_label = f"{away} @ {home}"

        for book in odds_data.get("bookmakers", []) or []:
            title = book.get("title") or book.get("key") or "Book"
            if title in EXCLUDED_BOOKS:
                continue
            book_name = (book.get("title") or book.get("key") or "Book").replace(" ", "")
            for market in book.get("markets", []) or []:
                mkey = market.get("key")
                outcomes = market.get("outcomes", []) or []

                for outcome in outcomes:
                    price = outcome.get("price")
                    if price is None:
                        continue

                    # Player field: many books use 'description' for player, 'name' for Over/Under/Yes/No
                    player = outcome.get("description") or outcome.get("name") or "Unknown"

                    if mkey == "player_anytime_td":
                        # Only "Yes" side for anytime TD
                        if (outcome.get("name") or "").strip().lower() != "yes":
                            continue
                        threshold = "Anytime"
                        group = "AnytimeTD"
                    else:
                        point = outcome.get("point")
                        if point is None:
                            continue
                        threshold = normalize_threshold(mkey, point)
                        if not threshold:
                            continue
                        group = next((g for g, k in MARKETS.items() if k == mkey), None)
                        if not group:
                            continue

                    rows.append({
                        "event": event_label,
                        "player": player,
                        "group": group,
                        "threshold": threshold,
                        "book": book_name,
                        "odds": price,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data fetched after processing odds (window ok, but no matching markets/thresholds).")
        sys.exit(0)

    # Wide table: one row per (event, player, group, threshold), per-book odds as columns
    pivot = df.pivot_table(
        index=["event", "player", "group", "threshold"],
        columns="book",
        values="odds",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    
    if pivot.empty:
        print("No data after pivot (no matching thresholds/books).")
        sys.exit(0)

    # Compute best price vs avg others in Decimal odds
    def compute_values(subdf: pd.DataFrame) -> pd.Series:
        keep_cols = [c for c in subdf.columns if c not in ["event", "player", "group", "threshold"]]
        odds_only = subdf[keep_cols]

        # Convert each cell element-wise to Decimal odds
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

    # Sort by value_ratio desc (best opportunities first)
    if "value_ratio" in out.columns:
        out = out.sort_values(by=["value_ratio"], ascending=[False], na_position="last")

    out.to_csv("nfl_player_props.csv", index=False)

    print("Saved nfl_player_props.csv with", len(out), "rows.")




