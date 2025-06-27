from scraper_dk import scrape_dk
from scraper_fd import scrape_fd
from scraper_mgm import scrape_mgm
from gpt_matcher import normalize_and_match
from app import run_app  
import pandas as pd
import streamlit.web.cli as stcli
import sys

def run_all_scrapers():
    all_results = []

    for scrape_func in [scrape_dk, scrape_fd, scrape_mgm]:
        try:
            results = scrape_func()
            if results:
                print(f"{scrape_func.__name__} returned {len(results)} rows.")
                all_results.extend(results)
            else:
                print(f"{scrape_func.__name__} returned no data.")
        except Exception as e:
            print(f"Error running {scrape_func.__name__}: {e}")

    if not all_results:
        print("❌ No data scraped.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    print(f"\n✅ Combined total: {len(df)} rows")
    return df

if __name__ == "__main__":
    df = run_all_scrapers()
    if df.empty:
        print("❌ No data to normalize.")
    else:
        matched_df = normalize_and_match(df)
        if matched_df is not None:
            # Save to CSV as backup; Streamlit can also use this if needed
            matched_df.to_csv("matched_output.csv", index=False)

            # Start the Streamlit app
            sys.argv = ["streamlit", "run", "app.py"]
            sys.exit(stcli.main())
