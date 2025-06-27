from playwright.sync_api import sync_playwright
import re
import time
#import pandas as pd

def parse_fanduel_hr_odds(text: str):
    lines = text.splitlines()
    parsed = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line or 'More wagers' in line or 'Show more' in line:
            i += 1
            continue

        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r"^[+-]\d+$", next_line):
                parsed.append({
                    "player": line,
                    "odds": next_line,
                    "bet_type": "to hit 1+ HR"
                })
                i += 2
                continue

        i += 1

    return parsed

def scrape_fd():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        print("🌐 Navigating to FanDuel MLB Parlay Builder...")
        page.goto("https://sportsbook.fanduel.com/navigation/mlb?tab=parlay-builder", timeout=60000)
        page.wait_for_timeout(5000)

        print("🎯 Clicking 'To Hit a Home Run Parlay Builder' tab...")
        try:
            page.locator("text=To Hit a Home Run Parlay Builder").first.click(timeout=5000)
            page.wait_for_timeout(5000)
        except:
            print("❌ Could not click HR parlay builder tab.")
            browser.close()
            return []

        print("🔄 Scrolling to bottom to load full page...")
        page.mouse.wheel(0, 10000)
        time.sleep(2)  # Let lazy content load

        # Step 1: Get full scroll height
        full_height = page.evaluate("() => document.body.scrollHeight")
        print(f"📏 Full page height: {full_height}px")

        # Step 2: Scroll back to top
        page.evaluate("() => window.scrollTo(0, 0)")
        time.sleep(0.5)

        # Step 3: Stepwise scroll and expand buttons
        steps = 20
        step_size = full_height // steps

        print("🔁 Beginning simple scroll-and-click loop...")

        for step in range(steps + 1):
            y_position = step * step_size
            print(f"🔽 Step {step}/{steps} — Scrolling to {y_position}px")
            page.evaluate(f"() => window.scrollTo(0, {y_position})")
            time.sleep(0.5)

            # Re-fetch buttons after each scroll
            buttons = page.locator("div[role='button'][aria-label='Show more']").all()

            for button in buttons:
                try:
                    if button.is_visible():
                        button.scroll_into_view_if_needed(timeout=1500)
                        button.click()
                        print("✅ Clicked: Show more")
                        time.sleep(0.2)
                except Exception as e:
                    print(f"⚠️ Failed to click button: {e}")



        print("📄 Extracting and parsing page text...\n")
        text = page.inner_text("body")
        browser.close()

        parsed = parse_fanduel_hr_odds(text)
        for item in parsed:
            item["book"] = "FanDuel"

        print(f"✅ FanDuel scraped {len(parsed)} props.")
        return parsed

# if __name__ == "__main__":
#     print("🔍 Testing FanDuel scraper directly...\n")
#     results = scrape_fd()
#     print(f"\n📝 Total props scraped: {len(results)}")

#     df = pd.DataFrame(results)
#     df.to_csv("fanduel_results.csv", index=False)
#     print("✅ Results saved to fanduel_results.csv")
