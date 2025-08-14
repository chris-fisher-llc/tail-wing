# daily_home_runs.py (drop-in replacement)
import requests
import csv
from datetime import datetime, timezone
from collections import defaultdict

API_KEY = "2e563a999b140389d269789871257204"  # use repo secret in CI; hardcode locally if you must
TARGET_MARKETS = ["batter_home_runs", "batter_home_runs_alternate"]

def _titlecase_book(book_key: str) -> str:
    # Known mappings for nicer column names; add more as needed
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

def get_today_events():
    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/events",
        params={"apiKey": API_KEY, "dateFormat": "iso"},
        timeout=30,
    )
    resp.raise_for_status()
    events = resp.json()
    today = datetime.now(timezone.utc).date()
    return [
        ev for ev in events
        if datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).date() == today
    ]

def get_home_run_odds(event_id):
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "oddsFormat": "american",
        "markets": ",".join(TARGET_MARKETS),
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def main():
    events = get_today_events()
    player_to_book_odds = defaultdict(dict)   # {player: {book_col: price}}
    books_seen = set()

    for ev in events:
        data = get_home_run_odds(ev["id"])
        if not data or "bookmakers" not in data:
            continue

        for book in data["bookmakers"]:
            bkey = book["key"]
            col_name = _titlecase_book(bkey)
            books_seen.add(col_name)

            for market in book.get("markets", []):
                if market.get("key") not in TARGET_MARKETS:
                    continue

                for outcome in market.get("outcomes", []):
                    # Keep Over 0.5 only; skip unders/no/alt that aren't 0.5
                    if (outcome.get("name","").lower() != "over") or (outcome.get("point") != 0.5):
                        continue

                    player = (outcome.get("description") or outcome.get("name") or "").strip()
                    if not player:
                        continue

                    price = outcome.get("price")
                    if price is None:
                        continue

                    # Save the best (highest) odds per player/book
                    if col_name not in player_to_book_odds[player] or price > player_to_book_odds[player][col_name]:
                        player_to_book_odds[player][col_name] = int(price)

    if not player_to_book_odds:
        print("No home run odds found for today.")
        return

    books_order = sorted(books_seen)  # stable columns for CSV

    rows = []
    for player, odds_by_book in sorted(player_to_book_odds.items()):
        # Compute value stats
        odds_list = [(book, odds) for book, odds in odds_by_book.items()]
        if not odds_list:
            continue

        # Best = highest American odds (larger numeric value)
        best_book, best_odds = max(odds_list, key=lambda x: x[1])

        other_odds = [o for b, o in odds_list if b != best_book]
        if len(other_odds) >= 1:
            avg_other = sum(other_odds) / len(other_odds)
            value = best_odds / avg_other if avg_other else ""
            value_flag = "TRUE" if (avg_other and value >= 1.10) else "FALSE"
        else:
            value = ""
            value_flag = "FALSE"

        # Build CSV row
        row = {
            "normalized_player": player,          # API data is already grouped/consistent
            "bet_type": "To Hit 1+ HR",
            "player": player,
            "value": round(value, 4) if value != "" else "",
            "value_book": best_book,
            "value_odds": best_odds,
            "value_flag": value_flag,
            "GPT Value": ""                       # reserved column; not used here
        }
        for book_col in books_order:
            row[book_col] = odds_by_book.get(book_col, "")

        rows.append(row)

    # Stable column order for Streamlit (dynamic odds columns in the middle)
    base_prefix = ["normalized_player", "bet_type", "player"]
    base_suffix = ["value", "value_book", "value_odds", "value_flag", "GPT Value"]
    fieldnames = base_prefix + books_order + base_suffix

    with open("matched_output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote matched_output.csv with {len(rows)} rows and {len(books_order)} book columns.")

if __name__ == "__main__":
    main()
