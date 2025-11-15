# =====================================================================
#   CLEAN CFB ODDS PULL SCRIPT (SPREADS • TOTALS • MONEYLINES)
#   Improvements:
#     • REGIONS now includes “us2” → pulls Hard Rock + ESPN Bet
#     • Clean sportsbook naming (no "_Odds" suffix)
#     • Simple mappings for all major US books
#     • Cleaner CSV output for Streamlit app
#     • One odds row = one book = one price per market
# =====================================================================

import os
import csv
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

API_KEY = "7f8cbb98207020adbd0218844a595725"
SPORT_KEY = "americanfootball_ncaaf"

TARGET_MARKETS = ["spreads", "totals", "h2h"]
REGIONS = "us,us2"             # ⭐ now includes HardRock + ESPNBet
ODDS_FORMAT = "american"
DATE_FMT = "iso"

ET = ZoneInfo("America/New_York")

# -----------------------------------------------------
# Clean sportsbook naming convention
# -----------------------------------------------------
BOOK_MAP = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "betrivers": "BetRivers",
    "caesars": "Caesars",
    "pointsbetus": "PointsBet",
    "espnbet": "ESPNBet",
    "hardrock": "HardRock",
    "wynnbet": "WynnBET",
    "bet365": "Bet365",
    "unibet_us": "Unibet",
    "bovada": "Bovada",
}

def map_book_name(raw):
    raw = raw.lower()
    return BOOK_MAP.get(raw, raw.title())   # fallback titlecase


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def american_to_decimal(american):
    try:
        a = float(american)
    except:
        return None
    if a > 0:
        return 1 + (a / 100.0)
    return 1 + (100.0 / abs(a))


def today_events():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events"
    r = requests.get(url, params={"apiKey": API_KEY, "dateFormat": DATE_FMT}, timeout=30)
    r.raise_for_status()

    events = r.json()
    today = datetime.now(ET).date()

    keep = []
    for ev in events:
        utc = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        et = utc.astimezone(ET)
        if et.date() == today:
            keep.append(ev)
    return keep


def event_odds(event_id):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "oddsFormat": ODDS_FORMAT,
        "markets": ",".join(TARGET_MARKETS),
        "dateFormat": DATE_FMT,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def iso_to_et(iso_ts):
    dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    return dt.astimezone(ET).strftime("%Y-%m-%d %H:%M ET")


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    events = today_events()
    if not events:
        print("No NCAAF events today.")
        return

    # Aggregation key: (event_id, bet_type, selection, line)
    agg = {}

    for ev in events:
        ev_id = ev["id"]
        home = ev["home_team"]
        away = ev["away_team"]
        game = f"{away} @ {home}"

        kickoff = iso_to_et(ev["commence_time"])

        odds_json = event_odds(ev_id)
        if "bookmakers" not in odds_json:
            continue

        for book in odds_json["bookmakers"]:
            book_key = map_book_name(book["key"])

            for market in book.get("markets", []):
                mkey = market["key"]

                if mkey not in TARGET_MARKETS:
                    continue

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "").strip()
                    price = outcome.get("price")
                    if price is None:
                        continue

                    # Normalize home/away
                    if name.lower() == "home":
                        name = home
                    elif name.lower() == "away":
                        name = away

                    if mkey == "spreads":
                        pt = outcome.get("point")
                        if pt is None:
                            continue
                        bet_type = "Spread"
                        selection = name
                        line = float(pt)

                    elif mkey == "totals":
                        pt = outcome.get("point")
                        lname = name.lower()
                        if pt is None or lname not in ("over", "under"):
                            continue
                        bet_type = "Total"
                        selection = "Over" if lname == "over" else "Under"
                        line = float(pt)

                    elif mkey == "h2h":
                        bet_type = "Moneyline"
                        if name not in (home, away):
                            continue
                        selection = name
                        line = ""   # moneyline has no line

                    else:
                        continue

                    opp = away if selection == home else home

                    key = (ev_id, bet_type, selection, line)

                    if key not in agg:
                        agg[key] = {
                            "game": game,
                            "kickoff_et": kickoff,
                            "bet_type": bet_type,
                            "selection": selection,
                            "opponent": opp,
                            "line": line if bet_type != "Moneyline" else "",
                            "books": {}
                        }

                    # Keep the best odds for this book
                    dec_now = american_to_decimal(price)
                    dec_prev = american_to_decimal(agg[key]["books"].get(book_key))

                    if dec_prev is None or (dec_now is not None and dec_now > dec_prev):
                        agg[key]["books"][book_key] = int(price)

    # -----------------------------------------------------
    # Build rows for CSV
    # -----------------------------------------------------
    all_books = set()
    for rec in agg.values():
        all_books |= set(rec["books"].keys())

    all_books = sorted(all_books)

    # Compute best_book and best_odds
    rows = []
    for (ev_id, bet_type, selection, line), rec in agg.items():
        books = rec["books"]

        # best offer
        best_book = None
        best_odds = None
        best_dec = -1

        for bk, odd in books.items():
            dec = american_to_decimal(odd)
            if dec is not None and dec > best_dec:
                best_dec = dec
                best_book = bk
                best_odds = odd

        # average other decimals
        others = [american_to_decimal(books[b]) for b in books if b != best_book]
        others = [x for x in others if x is not None]

        avg_other = sum(others) / len(others) if others else ""

        row = {
            "game": rec["game"],
            "kickoff_et": rec["kickoff_et"],
            "bet_type": rec["bet_type"],
            "selection": rec["selection"],
            "opponent": rec["opponent"],
            "line": rec["line"],
            "best_book": best_book,
            "best_odds": best_odds,
            "best_decimal": round(best_dec, 4) if isinstance(best_dec, float) else "",
            "avg_other_decimal": round(avg_other, 4) if isinstance(avg_other, float) else "",
        }

        for bk in all_books:
            row[bk] = books.get(bk, "")

        rows.append(row)

    # -----------------------------------------------------
    # Write CSV
    # -----------------------------------------------------
    script_dir = os.path.dirname(__file__)
    out_path = os.path.join(script_dir, "cfb_matched_output.csv")

    fieldnames = [
        "game", "kickoff_et", "bet_type", "selection", "opponent", "line",
        "best_book", "best_odds", "best_decimal", "avg_other_decimal"
    ] + list(all_books)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {out_path} with {len(rows)} rows and {len(all_books)} books.")


if __name__ == "__main__":
    main()
