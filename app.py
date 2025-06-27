import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="The Tail Wing", layout="wide")
st.markdown("<h1 style='text-align: center;'>✈️ The Tail Wing 💴</h1>", unsafe_allow_html=True)

def run_app(df=None):
    # Load DataFrame from CSV if none provided
    if df is None:
        try:
            df = pd.read_csv("matched_output.csv")
            modified_time = os.path.getmtime("matched_output.csv")
            formatted_time = datetime.fromtimestamp(modified_time).strftime("%I:%M %p EST on %B %d")
            st.markdown(f"**Odds last updated at {formatted_time}**")
        except FileNotFoundError:
            st.error("matched_output.csv not found.")
            return
    else:
        st.markdown("**Odds loaded from memory.**")

    if df.empty:
        st.warning("No data to display.")
        return

    # Rename and clean
    st.write("Columns before rename:", df.columns.tolist())
    df = df.rename(columns={
        "normalized_player": "Player",
        "bet_type": "Bet Type",
        "value": "Value",
        "value_book": "Best Book"
    })

    def to_american(x):
        try:
            x = int(float(x))
            return f"+{x}" if x > 0 else str(x)
        except:
            return ""

    for col in ["DraftKings_Odds", "FanDuel_Odds", "BetMGM_Odds"]:
        if col in df.columns:
            df[col] = df[col].apply(to_american)

    df["Value"] = df["Value"].astype(float).round(2).astype(str).str.rstrip("0").str.rstrip(".")
    df = df[["Player", "Bet Type", "DraftKings_Odds", "FanDuel_Odds", "BetMGM_Odds", "Value", "Best Book"]]

    # Filter
    with st.sidebar:
        st.header("Filter by Best Book")
        books = df["Best Book"].dropna().unique().tolist()
        selected_book = st.selectbox("", ["All"] + books)

    if selected_book != "All":
        df = df[df["Best Book"] == selected_book]

    df = df.sort_values("Value", ascending=False).reset_index(drop=True)

    # Gradient formatting
    def value_gradient(val):
        try:
            val = float(val)
            if val < 1.1:
                return "background-color: #f8d7da"
            green_scale = [
                "#e6f4ea", "#d4edda", "#b7e7bd", "#a8e6a1", "#8ae58a", "#6edc6e", "#5cd65c",
                "#47c947", "#33cc33", "#2eb82e", "#29a329", "#248f24", "#1f7a1f", "#1a661a",
                "#145214", "#0f3d0f", "#0a290a"
            ]
            index = min(int((val - 1.1) / 0.05), len(green_scale) - 1)
            return f"background-color: {green_scale[index]}; color: white" if index >= 8 else f"background-color: {green_scale[index]}"
        except:
            return ""

    def highlight_best_odds(row):
        style = [""] * len(row)
        best = row["Best Book"]
        gradient = value_gradient(row["Value"])
        for i, col in enumerate(row.index):
            if col == best:
                style[i] = gradient
        return style

    styled = df.style \
        .applymap(value_gradient, subset=["Value"]) \
        .apply(highlight_best_odds, axis=1) \
        .set_table_styles([
            {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center'), ('font-size', '16px')]}
        ])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

# Run if executed directly
if __name__ == "__main__":
    run_app()
