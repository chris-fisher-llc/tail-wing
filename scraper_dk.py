from playwright.sync_api import sync_playwright
import re
import json

def parse_dk_home_run_props(text: str) -> list[dict]:
    lines = text.splitlines()
    results = []
    current_player = None

    for i, line in enumerate(lines):
        line = line.strip()

        # Identify player names
        if "Total:" in line and i > 0:
            current_player = lines[i - 1].strip()

        # Identify HR lines
        if line.startswith("1+") and i + 1 < len(lines):
            odds_line = lines[i + 1].strip()
            if re.match(r"^[+-]\d+$", odds_line) and current_player:
                results.append({
                    "player": current_player,
                    "bet_type": "1+ HR",
                    "odds": odds_line
                })

    return results

def scrape_dk(bet_type: str = "home_run") -> list[dict]:
    if bet_type != "home_run":
        raise NotImplementedError("Only 'home_run' bet_type is supported.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        url = "https://sportsbook.draftkings.com/leagues/baseball/mlb?category=batter-props&subcategory=home-runs"
        page.goto(url, timeout=60000)
        page.wait_for_timeout(5000)

        text_content = page.inner_text("body")
        browser.close()

    parsed = parse_dk_home_run_props(text_content)
    for r in parsed:
        r["book"] = "DraftKings"
    
    print(f"✅ DraftKings scraped {len(parsed)} props.")
    return parsed

