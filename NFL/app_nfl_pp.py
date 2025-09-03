import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import pytz

st.set_page_config(page_title="The Tail Wing - NFL Player Props", layout="wide")

# ---- Header (no emojis) ----
st.markdown(
    """
    <h1 style='text-align: center; font-size: 42px;'>
        NFL Player Props — Anomaly Board
    </h1>
    <p style='text-align: center; font-size:18px; color: gray;'>
        Powered by The Tail Wing — scanning books for alt-yardage & anytime TD edges
    </p>
    """,
    unsafe_allow_html=True
)

# ---- CSV path resolution ----
def _find_csv_path() -> Path | None:
    # 1) Environment override
    env = os.getenv("NFL_PROPS_CSV")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 2) Common relative locations
    here = Path(__file__).resolve().parent
    candidates = [
        here / "nfl_player_props.csv",
        here / "nfl" / "nfl_player_props.csv",
        here.parent / "nfl_player_props.csv",
        here.parent / "nfl" / "nfl_player_props.csv",
        Path.cwd() / "nfl_player_props.csv",
        Path.cwd() / "nfl" / "nfl_player_props.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # 3) Last resort: recursive search near app (first match)
    try:
        for p in here.rglob("nfl_player_props.csv"):
            return p
    except Exception:
        pass
    return None


def run_app(df: pd.DataFrame | None = None):
    # Load DataFrame from CSV if none provided
    if df is None:
        csv_path = _find_csv_path()
        if not csv_path or not csv_path.exists():
            st.error(
                "nfl_player_props.csv not found.\n\n"
                "• Place the file next to this app, or in an 'nfl/' subfolder, or set env var NFL_PROPS_CSV to the full path.\n\n"
                f"Working directory: {Path.cwd()}\n"
            )
            return
        try:
            df = pd.read_csv(csv_path)
            # Clean up any stray 'Unnamed' columns created by CSV readers/pivots
            df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed')]
            df = df.dropna(axis=1, how="all")
            # "Last updated" in US/Eastern
            to_zone = pytz.timezone('US/Eastern')
            ts = datetime.fromtimestamp(csv_path.stat().st_mtime, pytz.utc)
            eastern = ts.astimezone(to_zone).strftime("%Y-%m-%d %I:%M %p %Z")
            st.caption(f"Odds last updated: {eastern}")
        except Exception as e:
            st.error(f"Error loading {csv_path}: {e}")
            st.caption("Odds last updated: {eastern}")
        except Exception as e:
            st.error(f"Error loading {csv_path}: {e}")
            return
    else:
        st.caption("Odds loaded from memory.")

    if df.empty:
        st.warning("No data to display.")
        return

    # --- Normalize column names ---
    rename_map = {
        "event": "Event",
        "player": "Player",
        "group": "Bet Type",
        "threshold": "Alt Line",
        "best_book": "Best Book",
        "best_odds": "Best Odds",
        "value_ratio": "Value",
    }
    df = df.rename(columns=rename_map)

    # Detect sportsbook odds columns dynamically (anything not in fixed set)
    fixed_cols = {"Event", "Player", "Bet Type", "Alt Line", "Best Book", "Best Odds", "Value",
                  "best_decimal", "avg_other"}
    odds_cols = [
        c for c in df.columns
        if c not in fixed_cols and not str(c).startswith("Unnamed")
    ]

    # Format odds nicely
    def to_american(x):
        try:
            x = int(float(x))
            return f"+{x}" if x > 0 else str(x)
        except Exception:
            return ""

    for col in odds_cols:
        df[col] = df[col].apply(to_american)
    if "Best Odds" in df.columns:
        df["Best Odds"] = df["Best Odds"].apply(to_american)

    # Format Value
    df["Value"] = pd.to_numeric(df.get("Value"), errors="coerce")
    df["_Value_print"] = df["Value"].map(lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if pd.notnull(x) else "")

    # Reorder columns for display
    display_cols = ["Event", "Player", "Bet Type", "Alt Line"] + odds_cols + ["Value", "_Value_print", "Best Book", "Best Odds"]
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols].copy()

    # --- Sidebar filters ---
    with st.sidebar:
        st.header("Filter by Best Book")
        books = df["Best Book"].dropna().unique().tolist() if "Best Book" in df.columns else []
        selected_book = st.selectbox("", ["All"] + sorted(books))

        st.header("Filter by Event")
        events = df["Event"].dropna().unique().tolist() if "Event" in df.columns else []
        selected_event = st.selectbox("", ["All"] + sorted(events))

        st.header("Filter by Bet Type")
        bet_types = df["Bet Type"].dropna().unique().tolist() if "Bet Type" in df.columns else []
        selected_bet_type = st.selectbox("", ["All"] + sorted(bet_types))

    if selected_book != "All":
        df = df[df["Best Book"] == selected_book]
    if selected_event != "All":
        df = df[df["Event"] == selected_event]
    if 'selected_bet_type' in locals() and selected_bet_type != "All":
        df = df[df["Bet Type"] == selected_bet_type]

        df = df[df["Event"] == selected_event]

    # --- Sort by Value ---
    if "Value" in df.columns:
        df = df.sort_values(by="Value", ascending=False, na_position="last")

    # Printable Value
    df["Value_display"] = df["_Value_print"]
    df.drop(columns=["_Value_print"], inplace=True)

    # -------- Styling --------
    # Step every 0.2 from 1.0 to 4.0; cap above 4.0; green gradient
    def value_step_style(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v <= 1.0:
            return ""
        capped = min(v, 4.0)
        step = int((capped - 1.0) // 0.2)
        step = max(0, min(step, 15))
        alpha = 0.12 + (0.95 - 0.12) * (step / 15.0)
        return f"background-color: rgba(34,139,34,{alpha}); font-weight: 600;"

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
        styled = styled.applymap(value_step_style, subset=["Value"])  # gradient fill by value
    styled = styled.apply(highlight_best_book_cells, axis=1).set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'),
                                     ('text-align', 'center'),
                                     ('font-size', '16px'),
                                     ('background-color', '#003366'),
                                     ('color', 'white')]}
    ])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

# Run if executed directly
if __name__ == "__main__":
    run_app()



