import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import pytz
import requests
import time

st.set_page_config(page_title="The Tail Wing - NBA Player Props (Free Board)", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; font-size: 42px;'>
       🏀 NBA Player Props — Free Board
    </h1>
    <p style='text-align: center; font-size:18px; color: gray;'>
        Scanning books for Points / Rebounds / Assists / 3PT / Steals / Blocks (alternate lines)
    </p>
    """,
    unsafe_allow_html=True
)

# ---------- GitHub Actions Trigger (manual refresh) ----------
def trigger_github_action():
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")  # e.g., "your-org-or-user/your-repo"
    workflow_file = st.secrets.get("GITHUB_WORKFLOW_FILE", "update-nba-player-props.yml")
    ref = st.secrets.get("GITHUB_REF", "main")

    if not token or not repo:
        st.error("Missing secrets: please set GITHUB_TOKEN and GITHUB_REPO in st.secrets.")
        return False

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"ref": ref}

    with st.spinner("Triggering GitHub Action…"):
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
    if resp.status_code == 204:
        st.success("Refresh kicked off. Odds will update automatically when the CSV is pushed.")
        return True
    else:
        st.error(f"Failed to trigger workflow ({resp.status_code}): {resp.text}")
        return False

def wait_for_csv_update(max_checks: int = 12, sleep_seconds: int = 10):
    csv_path = _find_csv_path()
    if not csv_path or not csv_path.exists():
        return
    old_mtime = csv_path.stat().st_mtime
    with st.spinner("Waiting for new data…"):
        for _ in range(max_checks):
            time.sleep(sleep_seconds)
            if not csv_path.exists():
                continue
            if csv_path.stat().st_mtime != old_mtime:
                st.success("Data updated — reloading!")
                st.rerun()

# ---------- CSV discovery ----------
def _find_csv_path() -> Path | None:
    env = os.getenv("NBA_PROPS_CSV")
    if env:
        p = Path(env)
        if p.exists():
            return p

    here = Path(__file__).resolve().parent
    candidates = [
        here / "nba_player_props.csv",
        here / "nba" / "nba_player_props.csv",
        here.parent / "nba_player_props.csv",
        here.parent / "nba" / "nba_player_props.csv",
        Path.cwd() / "nba_player_props.csv",
        Path.cwd() / "nba" / "nba_player_props.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    try:
        for p in here.rglob("nba_player_props.csv"):
            return p
    except Exception:
        pass
    return None

def to_american(x):
    try:
        x = int(float(x))
        return f"+{x}" if x > 0 else str(x)
    except Exception:
        return ""

# ---------- Top controls ----------
btn_cols = st.columns([1, 1, 1])
with btn_cols[1]:
    if st.button("Refresh Odds", use_container_width=True):
        if trigger_github_action():
            wait_for_csv_update()

def run_app(df: pd.DataFrame | None = None):
    if df is None:
        csv_path = _find_csv_path()
        if not csv_path or not csv_path.exists():
            st.error(
                "nba_player_props.csv not found.\n\n"
                "• Place the file next to this app, or in an 'nba/' subfolder, or set env var NBA_PROPS_CSV to the full path.\n\n"
                f"Working directory: {Path.cwd()}\n"
            )
            return
        try:
            df = pd.read_csv(csv_path)
            df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed')]
            df = df.dropna(axis=1, how="all")
            to_zone = pytz.timezone('US/Eastern')
            ts = datetime.fromtimestamp(csv_path.stat().st_mtime, pytz.utc)
            eastern = ts.astimezone(to_zone).strftime("%Y-%m-%d %I:%M %p %Z")
            st.caption(f"Odds last updated: {eastern}")
        except Exception as e:
            st.error(f"Error loading {csv_path}: {e}")
            return
    else:
        st.caption("Odds loaded from memory.")

    if df.empty:
        st.warning("No data to display.")
        return

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

    if "Bet Type" in df.columns:
        df["Bet Type"] = df["Bet Type"].astype(str).str.strip()

    fixed_cols = {"Event", "Player", "Bet Type", "Alt Line", "Best Book", "Best Odds", "Value",
                  "best_decimal", "avg_other"}
    odds_cols = [c for c in df.columns if c not in fixed_cols and not str(c).startswith("Unnamed")]

    for col in odds_cols:
        df[col] = df[col].apply(to_american)
    if "Best Odds" in df.columns:
        df["Best Odds"] = df["Best Odds"].apply(to_american)

    df["Value"] = pd.to_numeric(df.get("Value"), errors="coerce")
    df["_Value_print"] = df["Value"].map(lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if pd.notnull(x) else "")

    display_cols = ["Event", "Player", "Bet Type", "Alt Line"] + odds_cols + ["Value", "_Value_print", "Best Book", "Best Odds"]
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols].copy()

    # ---------- Sidebar Filters ----------
    with st.sidebar:
        st.header("Filters")
        events = ["All"] + sorted(df["Event"].dropna().unique().tolist()) if "Event" in df.columns else ["All"]
        selected_event = st.selectbox("Event", events, index=0)

        bet_types = ["All"] + sorted(df["Bet Type"].dropna().unique().tolist()) if "Bet Type" in df.columns else ["All"]
        selected_bet_type = st.selectbox("Bet Type", bet_types, index=0)

        books = ["All"] + sorted(df["Best Book"].dropna().unique().tolist()) if "Best Book" in df.columns else ["All"]
        selected_book = st.selectbox("Best Book", books, index=0)

    if selected_event != "All":
        df = df[df["Event"] == selected_event]
    if selected_bet_type != "All":
        df = df[df["Bet Type"].astype(str).str.strip() == selected_bet_type]
    if selected_book != "All":
        df = df[df["Best Book"] == selected_book]

    if "Value" in df.columns:
        df = df.sort_values(by="Value", ascending=False, na_position="last")

    df["Value_display"] = df["_Value_print"]
    df.drop(columns=["_Value_print"], inplace=True)

    # ---------- Styling ----------
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
        styled = styled.applymap(value_step_style, subset=["Value"])
    styled = styled.apply(highlight_best_book_cells, axis=1).set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'),
                                     ('text-align', 'center'),
                                     ('font-size', '16px'),
                                     ('background-color', '#003366'),
                                     ('color', 'white')]}
    ])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

if __name__ == "__main__":
    run_app()
