# scraper_mgm.py

from playwright.sync_api import sync_playwright
import re

def scrape_mgm():
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(permissions=[], geolocation=None)
        page = context.new_page()

        print("\n🌐 Navigating to BetMGM MLB page...")
        page.goto("https://www.ny.betmgm.com/en/sports/baseball-23", timeout=60000)
        page.wait_for_timeout(5000)

        try:
            page.keyboard.press("Escape")
            print("🛑 Location popup dismissed.")
        except:
            print("⚠️ Could not dismiss popup.")

        page.wait_for_selector("a.grid-info-wrapper")
        game_links = page.query_selector_all("a.grid-info-wrapper")
        game_urls = list({link.get_attribute("href") for link in game_links if link.get_attribute("href")})

        print(f"📋 Found {len(game_urls)} games.")
        for i, url in enumerate(game_urls):
            full_url = f"https://www.ny.betmgm.com{url}"
            print(f"\n➡️ Game {i+1}/{len(game_urls)}: {full_url}")
            game_page = context.new_page()
            game_page.goto(full_url, timeout=4000)
            game_page.wait_for_timeout(4000)

            try:
                hr_tab = game_page.get_by_text("Batter Home Runs", exact=False).first
                hr_tab.click()
                game_page.wait_for_timeout(2000)

                try:
                    game_page.get_by_text("Show More", exact=False).first.click(timeout=2000)
                    print("🔽 Show More clicked.")
                    game_page.wait_for_timeout(1000)
                except:
                    print("ℹ️ No Show More button found.")
            except:
                print("❌ Could not find Batter Home Runs tab.")
                game_page.close()
                continue

            # Scrape structured DOM instead of raw text
            try:
                player_elements = game_page.query_selector_all(".player-props-player-name")
                for player_elem in player_elements:
                    player = player_elem.inner_text().strip()
                    odds_block = player_elem.evaluate_handle("node => node.parentElement.parentElement")
                    over_selector = odds_block.query_selector(".option .name")
                    if over_selector and "O 0.5" in over_selector.inner_text():
                        value_elem = odds_block.query_selector(".option .value span")
                        if value_elem:
                            odds = value_elem.inner_text().strip()
                            results.append({
                                "player": player,
                                "odds": odds,
                                "bet_type": "O 0.5 HR",
                                "book": "BetMGM"
                            })
                            print(player, odds)
            except Exception as e:
                print(f"⚠️ Parsing error: {e}")

            game_page.close()

    print(f"✅ BetMGM scraped {len(results)} props.")
    return results

