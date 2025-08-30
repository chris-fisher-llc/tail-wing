# daily_cfb_spreads_totals.py
import os
import csv
import math
import requests
from datetime import datetime, timezone
from collections import defaultdict
from zoneinfo import ZoneInfo  # Python 3.9+

#API_KEY = os.getenv("THE_ODDS_API_KEY", "YOUR_API_KEY_HERE")
API_KEY = "7f8cbb98207020adbd0218844a595725"
SPORT_KEY = "americanfootball_ncaaf"
TARGET_MARKETS = ["spreads", "totals"]
REGIONS = "us"
ODDS_FORMAT = "american"
DATE_FMT = "iso"

ET = ZoneInfo("America/New_York")

def _titlecase_book(book_key: str) -> str:
    mapping = {
        "draftkings": "DraftKings",
        "fanduel": "FanDuel",
        "betmgm": "BetMGM",
        "betrivers": "BetRivers",
        "caesars": "Caesars",
        "pointsbetus": "PointsBet",
        "barstool": "Barstool",
        "espnbet": "ESPNBet",
        "wynnbet": "WynnBET",
        "bet365": "Bet365",
        "unibet_us": "Unibet",
    }
    label = mapping.get(book_key, book_key.replace("_", " ").title().replace(" ", ""))
    return f"{label}_Odds"

def _american_to_decimal(american):
    if american is None or american == "":
        return None
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a > 0:
        return 1.0 + (a / 100.0)
    if a < 0:
        return 1.0 + (100.0 / abs(a))
    return None

def _today_events():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events"
    resp = requests.get(url, params={"apiKey": API_KEY, "dateFormat": DATE_FMT}, timeout=30)
    resp.raise_for_status()
    events = resp.json()
    today = datetime.now(timezone.utc).date()
    return [
        ev for ev in events
        if datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).date() == today
    ]

def _event_odds(event_id: str):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "oddsFormat": ODDS_FORMAT,
        "markets": ",".join(TARGET_MARKETS),
        "dateFormat": DATE_FMT,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def _iso_to_et_str(iso_ts: str) -> str:
    dt_utc = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    dt_et = dt_utc.astimezone(ET)
    return dt_et.strftime("%Y-%m-%d %H:%M ET")

def main():
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        raise SystemExit("Set THE_ODDS_API_KEY env var or edit API_KEY in the script.")

    events = _today_events()
    if not events:
        print("No NCAAF events for today.")
        return

    # bet_key = (event_id, bet_type, selection, line_point)
    agg = {}
    books_seen = set()

    for ev in events:
        ev_id = ev["id"]
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        game_label = f"{away} @ {home}"
        kickoff_iso = ev["commence_time"]
        kickoff_et = _iso_to_et_str(kickoff_iso)

        odds_payload = _event_odds(ev_id)
        if not odds_payload or "bookmakers" not in odds_payload:
            continue

        for book in odds_payload["bookmakers"]:
            bkey = book["key"]
            book_col = _titlecase_book(bkey)
            books_seen.add(book_col)

            for market in book.get("markets", []):
                mkey = market.get("key")
                if mkey not in TARGET_MARKETS:
                    continue

                for outcome in market.get("outcomes", []):
                    name = (outcome.get("name") or "").strip()
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if price is None or point is None:
                        continue

                    if mkey == "spreads":
                        if name not in (home, away):
                            if name.lower() == "home":
                                name = home
                            elif name.lower() == "away":
                                name = away
                            else:
                                continue
                        bet_type = "Spread"
                        selection = name
                    elif mkey == "totals":
                        lname = name.lower()
                        if lname not in ("over", "under"):
                            continue
                        bet_type = "Total"
                        selection = "Over" if lname == "over" else "Under"
                    else:
                        continue

                    bet_key = (ev_id, bet_type, selection, float(point))
                    if bet_key not in agg:
                        agg[bet_key] = {
                            "event_id": ev_id,
                            "game": game_label,
                            "home": home,
                            "away": away,
                            "kickoff_et": kickoff_et,
                            "bet_type": bet_type,
                            "selection": selection,
                            "point": float(point),
                            "opponent": away if selection == home else (home if selection == away else ""),
                            "book_odds": {},
                        }

                    current = agg[bet_key]["book_odds"].get(book_col)
                    if current is None:
                        take_it = True
                    else:
                        cur_dec = _american_to_decimal(current) or -math.inf
                        new_dec = _american_to_decimal(price) or -math.inf
                        take_it = new_dec > cur_dec

                    if take_it:
                        agg[bet_key]["book_odds"][book_col] = int(price)

    if not agg:
        print("No spreads/totals found for today.")
        return

    books_order = sorted(books_seen)

    rows = []
    for (ev_id, bet_type, selection, point), rec in sorted(agg.items(), key=lambda x: (x[1]["game"], x[1]["bet_type"], x[1]["selection"], x[1]["point"])):
        odds_by_book = rec["book_odds"]
        if not odds_by_book:
            continue

        odds_list = [(book, odds) for book, odds in odds_by_book.items()]
        best_book, best_odds = max(
            odds_list,
            key=lambda bo: (_american_to_decimal(bo[1]) or -math.inf)
        )
        decs = [(_american_to_decimal(o) or None) for _, o in odds_list]
        other_decs = [d for (b, o), d in zip(odds_list, decs) if b != best_book and d is not None]
        best_dec = _american_to_decimal(best_odds)
        if best_dec is not None and other_decs:
            avg_other_dec = sum(other_decs) / len(other_decs)
            value_ratio = best_dec / avg_other_dec if avg_other_dec else ""
            value_flag = "TRUE" if (isinstance(value_ratio, float) and value_ratio >= 1.02) else "FALSE"
        else:
            avg_other_dec = ""
            value_ratio = ""
            value_flag = "FALSE"

        row = {
            "game": rec["game"],
            "kickoff_et": rec["kickoff_et"],
            "bet_type": rec["bet_type"],
            "selection": rec["selection"],
            "opponent": rec["opponent"],
            "line": rec["point"],
            "best_book": best_book,
            "best_odds": best_odds,
            "best_decimal": round(best_dec, 4) if isinstance(best_dec, float) else "",
            "avg_other_decimal": round(avg_other_dec, 4) if isinstance(avg_other_dec, float) else "",
            "value_ratio": round(value_ratio, 4) if isinstance(value_ratio, float) else "",
            "value_flag": value_flag,
        }

        for book_col in books_order:
            row[book_col] = odds_by_book.get(book_col, "")

        rows.append(row)

    prefix = ["game", "kickoff_et", "bet_type", "selection", "opponent", "line"]
    suffix = ["best_book", "best_odds", "best_decimal", "avg_other_decimal", "value_ratio", "value_flag"]
    fieldnames = prefix + books_order + suffix

    # Save inside the same folder as this script (CFB/)
    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, "cfb_matched_output.csv")
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote {output_path} with {len(rows)} rows and {len(books_order)} book columns.")

if __name__ == "__main__":
    main()

