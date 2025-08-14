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
            # "Last updated" in US/Eastern
            to_zone = pytz.timezone('US/Eastern')
            ts = datetime.fromtimestamp(os.path.getmtime("matched_output.csv"), pytz.utc)
            eastern = ts.astimezone(to_zone).strftime("%Y-%m-%d %I:%M %p %Z")
            st.caption(f"Odds last updated: {eastern}")
        except FileNotFoundError:
            st.error("matched_output.csv not found.")
            return
        except Exception as e:
            st.error(f"Error loading matched_output.csv: {e}")
            return
    else:
        st.caption("Odds loaded from memory.")

    if df.empty:
        st.warning("No data to display.")
        return

    # --- Normalize column names used in UI ---
    df = df.rename(columns={
        "normalized_player": "Player",
        "bet_type": "Bet Type",
        "value": "Value",
        "value_book": "Best Book"
    })

    # Detect all sportsbook odds columns dynamically (e.g., DraftKings_Odds, FanDuel_Odds, etc.)
    odds_cols = sorted([c for c in df.columns if c.endswith("_Odds")])

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
    # Keep a touch of precision for sorting, then pretty print
    df["_Value_print"] = df["Value"].map(lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if pd.notnull(x) else "")
    # Keep numeric Value for sorting/styling

    # Reorder columns for display (retain _Value_print so we can turn it into Value_display later)
    display_cols = ["Player", "Bet Type"] + odds_cols + ["Value", "_Value_print", "Best Book"]
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols].copy()

    # --- Best Book filter (replicates original UX) ---
    with st.sidebar:
        st.header("Filter by Best Book")
        books = df["Best Book"].dropna().unique().tolist()
        selected_book = st.selectbox("", ["All"] + sorted(books))
    if selected_book != "All":
        df = df[df["Best Book"] == selected_book]

    # --- Default sort by Value descending ---
    df = df.sort_values(by="Value", ascending=False, na_position="last")

    # Swap Value to printable string for rendering while preserving numeric for styling
    df["Value_display"] = df["_Value_print"]
    df.drop(columns=["_Value_print"], inplace=True)

    # -------- Styling helpers --------
    # Step gradient every 0.1 between 1.1 and 2.5 (15 steps: 1.1..2.5 inclusive)
    def value_step_style(val):
        try:
            v = float(val)
        except:
            return ""
        if v < 1.1:
            return ""
        # index 0 for [1.1,1.2), 1 for [1.2,1.3), ... up to cap at 2.5
        capped = min(v, 2.5)
        step = int((capped - 1.1) // 0.1)
        step = max(0, min(step, 14))
        # alpha from 0.12 → 0.90 across 15 steps
        alpha = 0.12 + (0.90 - 0.12) * (step / 14.0)
        return f"background-color: rgba(0,128,0,{alpha}); font-weight: 600;"

    # Highlight the row's Best Book column cell using the same value-based shade
    def highlight_best_book_cells(row):
        styles = [""] * len(row)
        best = row.get("Best Book", "")
        # compute shade based on numeric Value
        shade = value_step_style(row.get("Value", ""))
        if best and best in row.index:
            idx = list(row.index).index(best)
            styles[idx] = shade
        return styles

    # Build styled DataFrame:
    # Use "Value_display" for showing; use numeric "Value" for shade calcs (parsed inside value_step_style).
    render_df = df.copy()
    render_df["Value"] = render_df["Value_display"]
    render_df.drop(columns=["Value_display"], inplace=True)

    styled = render_df.style

    if "Value" in render_df.columns:
        styled = styled.applymap(value_step_style, subset=["Value"])

    styled = styled.apply(highlight_best_book_cells, axis=1).set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'),
                                     ('text-align', 'center'),
                                     ('font-size', '16px')]}
    ])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

# Run if executed directly
if __name__ == "__main__":
    run_app()
