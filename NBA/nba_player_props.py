import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

################################################################################
# NBA Player Props Odds Scraper (Tail Wing)
# - Targets Points, Rebounds, Assists, 3PT Made, Steals, Blocks at fixed thresholds
# - Compares best price vs average of others (in Decimal odds)
# - Time window: from NOW through end of TOMORROW in US/Eastern (inclusive)
# - Outputs: nba_player_props.csv
################################################################################

# API key & base
API_KEY = os.getenv("THE_ODDS_API_KEY", "7f8cbb98207020adbd0218844a595725")
API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"

# ---- Config: thresholds & markets -------------------------------------------
THRESHOLDS = {
    "Points":   [10, 15, 20, 25, 30, 35, 40, 45, 50],
    "Rebounds": [4, 6, 8, 10, 12, 14, 16],
    "Assists":  [4, 6, 8, 10, 12, 14, 16],
    "Threes":   [1, 2, 3, 4, 5, 6, 7],
    "Steals":   [1, 2, 3, 4, 5],
    "Blocks":   [1, 2, 3, 4, 5],
}

# The Odds API market keys vary by book/sport; include fallbacks.
# We'll request all of these; whichever appear will be normalized into our groups.
MARKETS = {
    "Points":   ["player_points_alternate", "player_points"],
    "Rebounds": ["player_rebounds_alternate", "player_rebounds"],
    "Assists":  ["player_assists_alternate", "player_assists"],
    "Threes":   ["player_threes_made_alternate", "player_threes_made", "player_threes", "player_3pt_made", "player_3pt_made_alternate"],
    "Steals":   ["player_steals_alternate", "player_steals"],
    "Blocks":   ["player_blocks_alternate", "player_blocks"],
}

# Books to exclude (correlated with NFL script)
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


def nb_window_now_to_tomorrow():
    """Return (start_utc, end_utc) covering from 'now' through end of *tomorrow* in ET."""
    tz_et = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(tz_et)

    # End at 23:59:59 ET tomorrow
    tomorrow_et = (now_et + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    end_utc = tomorrow_et.astimezone(timezone.utc)
    return now_utc, end_utc


# Build a reverse map from market_key -> group for quick lookup
MARKET_TO_GROUP = {}
for group, keys in MARKETS.items():
    for k in keys:
        MARKET_TO_GROUP[k] = group

# Single string with *unique* market keys to query
REQUEST_MARKETS = ",".join(sorted({k for keys in MARKETS.values() for k in keys}))

def normalize_threshold(market_key: str, point: float):
    """Map an alt-line 'point' to one of our canonical thresholds with +/-0.5 tolerance."""
    group = MARKET_TO_GROUP.get(market_key)
    if not group:
        return None
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


# ---- API calls ---------------------------------------------------------------

def fetch_events():
    """Fetch NBA events and filter to [now, end-of-tomorrow ET] window in UTC."""
    if not API_KEY:
        print("ERROR: THE_ODDS_API_KEY is not set in your environment.")
        sys.exit(1)

    start_utc, end_utc = nb_window_now_to_tomorrow()

    url = f\"{API_BASE}/sports/{SPORT_KEY}/events\"
    params = {\"apiKey\": API_KEY, \"dateFormat\": \"iso\"}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    events = resp.json() or []

    out = []
    for e in events:
        # commence_time is ISO 8601 in UTC with 'Z'
        dt_utc = datetime.fromisoformat(e[\"commence_time\"].replace(\"Z\", \"+00:00\"))
        if start_utc <= dt_utc <= end_utc:
            out.append(e)
    return out


def fetch_event_odds(event_id: str):
    url = f\"{API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds\"
    params = {
        \"apiKey\": API_KEY,
        \"regions\": \"us,us2\",
        \"markets\": REQUEST_MARKETS,
        \"oddsFormat\": \"american\",
        \"dateFormat\": \"iso\",
    }
    resp = requests.get(url, params=params, timeout=25)
    resp.raise_for_status()
    return resp.json()


# ---- Main --------------------------------------------------------------------

if __name__ == \"__main__\":
    events = fetch_events()
    if not events:
        print(\"No NBA events found in the now→tomorrow window.\")
        sys.exit(0)

    rows = []

    for ev in events:
        odds_data = fetch_event_odds(ev[\"id\"]) or {}
        home, away = ev.get(\"home_team\"), ev.get(\"away_team\")
        event_label = f\"{away} @ {home}\"

        for book in odds_data.get(\"bookmakers\", []) or []:
            title = book.get(\"title\") or book.get(\"key\") or \"Book\"
            if title in EXCLUDED_BOOKS:
                continue
            book_name = (book.get(\"title\") or book.get(\"key\") or \"Book\").replace(\" \", \"\")
            for market in book.get(\"markets\", []) or []:
                mkey = market.get(\"key\")
                group = MARKET_TO_GROUP.get(mkey)
                if not group:
                    continue  # not one of our targeted markets

                outcomes = market.get(\"outcomes\", []) or []
                for outcome in outcomes:
                    price = outcome.get(\"price\")
                    if price is None:
                        continue

                    # Player: often in 'description'; fallback to 'name'
                    player = outcome.get(\"description\") or outcome.get(\"name\") or \"Unknown\"

                    point = outcome.get(\"point\")
                    thr = normalize_threshold(mkey, point)
                    if not thr:
                        continue

                    rows.append({
                        \"event\": event_label,
                        \"player\": player,
                        \"group\": group,
                        \"threshold\": thr,
                        \"book\": book_name,
                        \"odds\": price,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print(\"No NBA data fetched after processing odds (window ok, but no matching markets/thresholds).\");
        sys.exit(0)

    # Wide table: one row per (event, player, group, threshold), per-book odds as columns
    pivot = df.pivot_table(
        index=[\"event\", \"player\", \"group\", \"threshold\"],
        columns=\"book\",
        values=\"odds\",
        aggfunc=\"first\",
    ).reset_index()
    pivot.columns.name = None

    if pivot.empty:
        print(\"No NBA data after pivot (no matching thresholds/books).\");
        sys.exit(0)

    # Compute best price vs avg others in Decimal odds
    def compute_values(subdf: pd.DataFrame) -> pd.Series:
        keep_cols = [c for c in subdf.columns if c not in [\"event\", \"player\", \"group\", \"threshold\"]]
        odds_only = subdf[keep_cols]

        # Convert each cell element-wise to Decimal odds
        decs = odds_only.map(american_to_decimal_one)
        row_decs = decs.iloc[0].dropna()
        if row_decs.empty:
            return pd.Series({
                \"best_book\": None,
                \"best_odds\": None,
                \"best_decimal\": None,
                \"avg_other\": None,
                \"value_ratio\": None,
            })

        best_col = row_decs.idxmax()
        best_val = float(row_decs.max())
        avg_other = float(row_decs.drop(labels=[best_col]).mean()) if row_decs.size > 1 else None
        ratio = (best_val / avg_other) if avg_other else None
        best_odds = odds_only.iloc[0][best_col]

        return pd.Series({
            \"best_book\": best_col,
            \"best_odds\": best_odds,
            \"best_decimal\": best_val,
            \"avg_other\": avg_other,
            \"value_ratio\": ratio,
        })

    metrics = pivot.groupby([\"event\", \"player\", \"group\", \"threshold\"], as_index=False).apply(compute_values).reset_index(drop=True)
    out = pd.merge(pivot, metrics, on=[\"event\", \"player\", \"group\", \"threshold\"], how=\"left\")


    # Sort by value_ratio desc (best opportunities first)
    if \"value_ratio\" in out.columns:
        out = out.sort_values(by=[\"value_ratio\"], ascending=[False], na_position=\"last\")


    out.to_csv(\"nba_player_props.csv\", index=False)
    print(\"Saved nba_player_props.csv with\", len(out), \"rows.\")
