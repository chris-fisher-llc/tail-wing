import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pytz

st.set_page_config(page_title="The Tail Wing", layout="wide")
st.markdown("<h1 style='text-align: center;'>✈️ The Tail Wing 💴</h1>", unsafe_allow_html=True)

def run_app(df=None):
    # Load DataFrame from CSV if none provided
    if df is None:
        try:
            df = pd.read_csv("matched_output.csv")
            from_zone = pytz.utc
            to_zone = pytz.timezone('US/Eastern')
            # Try to detect a commit or file timestamp for "Last updated"
            ts = None
            try:
                # If running on GH Pages / raw file, fallback to local mtime
                ts = datetime.fromtimestamp(os.path.getmtime("matched_output.csv"), pytz.utc)
            except Exception:
                pass
            if ts:
                eastern = ts.astimezone(to_zone).strftime("%Y-%m-%d %I:%M %p %Z")
                st.caption(f"Last updated: {eastern}")
        except FileNotFoundError:
            st.warning("No matched_output.csv found yet. Once the workflow runs, this will populate.")
            return
        except Exception as e:
            st.error(f"Error loading matched_output.csv: {e}")
            return

    if df.empty:
        st.info("No rows to display yet.")
        return

    # --- Normalize column names used in UI ---
    df = df.rename(columns={
        "normalized_player": "Player",
        "bet_type": "Bet Type",
        "value": "Value",
        "value_book": "Best Book"
    })

    # Detect all sportsbook odds columns dynamically (e.g., DraftKings_Odds, FanDuel_Odds, etc.)
    odds_cols = [c for c in df.columns if c.endswith("_Odds")]

    # Safely format odds to American style strings
    def to_american(x):
        try:
            x = int(float(x))
            return f"+{x}" if x > 0 else str(x)
        except:
            return ""

    for col in odds_cols:
        df[col] = df[col].apply(to_american)

    # Clean and format Value column safely
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Value"] = df["Value"].round(3)
    df["Value"] = df["Value"].map(lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if pd.notnull(x) else "")

    # Reorder columns for display
    display_cols = ["Player", "Bet Type"] + sorted(odds_cols) + ["Value", "Best Book"]
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols].copy()

    # --- Styling helpers ---
    # Highlight the best book column for each row (column name matches Best Book value, e.g., "FanDuel_Odds")
    def highlight_best_odds(row):
        styles = [""] * len(row)
        if "Best Book" in row.index and pd.notnull(row["Best Book"]):
            best_col = str(row["Best Book"])
            if best_col in row.index:
                idx = list(row.index).index(best_col)
                styles[idx] = "font-weight: 600; background-color: rgba(0, 128, 0, 0.12);"
        return styles

    # Shade Value cell by magnitude (>=1.10 green, >=1.05 lighter green)
    def value_shade(val):
        try:
            v = float(val)
        except:
            return ""
        if v >= 1.10:
            return "background-color: rgba(0, 128, 0, 0.22); font-weight: 600;"
        if v >= 1.05:
            return "background-color: rgba(0, 128, 0, 0.12);"
        return ""

    # Build styled DataFrame
    styled = df.style

    if "Value" in df.columns:
        styled = styled.applymap(value_shade, subset=["Value"])

    styled = styled.apply(highlight_best_odds, axis=1).set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'),
                                     ('text-align', 'center'),
                                     ('font-size', '16px')]}
    ])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

# Run if executed directly
if __name__ == "__main__":
    run_app()
