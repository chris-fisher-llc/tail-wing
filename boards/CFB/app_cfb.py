import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pytz

st.set_page_config(page_title="The Tail Wing - CFB", layout="wide")

# Themed header
st.markdown(
    """
    <h1 style='text-align: center; font-size: 42px;'>
        🏈 College Football Anomaly Board 🎓
    </h1>
    <p style='text-align: center; font-size:18px; color: gray;'>
        Powered by The Tail Wing &mdash; odds scanning across books for spreads and totals
    </p>
    """,
    unsafe_allow_html=True
)

def run_app(df=None):
    # Load DataFrame from CSV if none provided
    if df is None:
        try:
            script_dir = os.path.dirname(__file__)
            csv_path = os.path.join(script_dir, "cfb_matched_output.csv")           
            df = pd.read_csv(csv_path)

            # "Last updated" in US/Eastern
            to_zone = pytz.timezone('US/Eastern')
            ts = datetime.fromtimestamp(os.path.getmtime(csv_path), pytz.utc)
            eastern = ts.astimezone(to_zone).strftime("%Y-%m-%d %I:%M %p %Z")
            st.caption(f"📅 Odds last updated: {eastern}")
        except FileNotFoundError:
            st.error("cfb_matched_output.csv not found.")
            return
        except Exception as e:
            st.error(f"Error loading cfb_matched_output.csv: {e}")
            return
    else:
        st.caption("Odds loaded from memory.")

    if df.empty:
        st.warning("No data to display.")
        return

    # --- Normalize column names ---
    df = df.rename(columns={
        "game": "Game",
        "kickoff_et": "Kickoff",
        "bet_type": "Bet Type",
        "selection": "Selection",
        "line": "Line",
        "value_ratio": "Value",
        "best_book": "Best Book",
        "best_odds": "Best Odds"
    })

    # Format Kickoff column → "12:00 p.m. ET"
    if "Kickoff" in df.columns:
        df["Kickoff"] = pd.to_datetime(df["Kickoff"], errors="coerce").dt.strftime("%-I:%M %p ET")
        df["Kickoff"] = df["Kickoff"].str.replace("AM", "a.m.", regex=False)
        df["Kickoff"] = df["Kickoff"].str.replace("PM", "p.m.", regex=False)

    # Format Line column → always 1 decimal place
    if "Line" in df.columns:
        df["Line"] = pd.to_numeric(df["Line"], errors="coerce").map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")

    # Detect sportsbook odds columns dynamically
    odds_cols = sorted([c for c in df.columns if c.endswith("_Odds")])

    # Format odds nicely
    def to_american(x):
        try:
            x = int(float(x))
            return f"+{x}" if x > 0 else str(x)
        except:
            return ""

    for col in odds_cols:
        df[col] = df[col].apply(to_american)

    # Format Value
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["_Value_print"] = df["Value"].map(lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if pd.notnull(x) else "")

    # Reorder columns
    display_cols = ["Game", "Kickoff", "Bet Type", "Selection", "Line"] + odds_cols + ["Value", "_Value_print", "Best Book", "Best Odds"]
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols].copy()

    # --- Sidebar filter ---
    with st.sidebar:
        st.header("📚 Filter by Best Book")
        books = df["Best Book"].dropna().unique().tolist()
        selected_book = st.selectbox("", ["All"] + sorted(books))
    if selected_book != "All":
        df = df[df["Best Book"] == selected_book]

    # --- Sort by Value ---
    df = df.sort_values(by="Value", ascending=False, na_position="last")

    # Printable Value
    df["Value_display"] = df["_Value_print"]
    df.drop(columns=["_Value_print"], inplace=True)

    # -------- Styling --------
    # Gradient every 0.025 between 1.0 and 1.35
    def value_step_style(val):
        try:
            v = float(val)
        except:
            return ""
        if v <= 1.0:
            return ""
        capped = min(v, 1.35)
        step = int((capped - 1.0) // 0.025)
        step = max(0, min(step, 14))
        alpha = 0.12 + (0.90 - 0.12) * (step / 14.0)
        return f"background-color: rgba(34,139,34,{alpha}); font-weight: 600;"  # forest green, CFB vibe

    def highlight_best_book_cells(row):
        styles = [""] * len(row)
        best = row.get("Best Book", "")
        shade = value_step_style(row.get("Value", ""))
        if best and best in row.index:
            idx = list(row.index).index(best)
            styles[idx] = shade
        return styles

    render_df = df.copy()
    render_df["Value"] = render_df["Value_display"]
    render_df.drop(columns=["Value_display"], inplace=True)

    styled = render_df.style
    if "Value" in render_df.columns:
        styled = styled.applymap(value_step_style, subset=["Value"])
    styled = styled.apply(highlight_best_book_cells, axis=1).set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'),
                                     ('text-align', 'center'),
                                     ('font-size', '16px'),
                                     ('background-color', '#800000'),   # maroon headers
                                     ('color', 'white')]}
    ])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

# Run if executed directly
if __name__ == "__main__":
    run_app()
