import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import pytz
import requests
import time
import math
import streamlit.components.v1 as components

st.set_page_config(page_title="The Tail Wing - NBA Player Props (Free Board)", layout="wide")

# --- Auto-clear cache each run (fixes mobile stuck sessions) ---
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass

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

# --- Optional: detect mobile (for compact layout) ---
components.html(
    """
    <script>
      const w = Math.min(window.innerWidth || 9999, screen.width || 9999);
      const isMobile = w < 800;
      var streamlitDoc = window.parent;
      streamlitDoc.postMessage({isMobile:isMobile, type: 'TAILWING_MOBILE_FLAG'}, "*");
    </script>
    """,
    height=0,
)

# ---------- GitHub Actions Trigger ----------
def trigger_github_action():
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")
    workflow_file = st.secrets.get("GITHUB_WORKFLOW_FILE", "main_nba.yml")
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

# ---------- Refresh button ----------
btn_cols = st.columns([1, 1, 1])
with btn_cols[1]:
    if st.button("Refresh Odds", use_container_width=True):
        if trigger_github_action():
            wait_for_csv_update()

# ---------- Core ----------
def run_app(df: pd.DataFrame | None = None):
    if df is None:
        csv_path = _find_csv_path()
        if not csv_path or not csv_path.exists():
            st.error("nba_player_props.csv not found.")
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
            st.error(f"Error loading CSV: {e}")
            return

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
        "value_ratio": "% Over Market Avg",
    }
    df = df.rename(columns=rename_map)

    # Identify sportsbook columns
    fixed_cols_raw = {
        "Event", "Player", "Bet Type", "Alt Line", "Best Book", "Best Odds",
        "% Over Market Avg", "best_decimal", "avg_other"
    }
    all_cols = list(df.columns)
    book_cols = [c for c in all_cols if c not in fixed_cols_raw and not str(c).startswith("Unnamed")]

    # Books available
    def _is_valid_num(v):
        try:
            return pd.notnull(v) and str(v).strip() != ""
        except Exception:
            return False
    df["# Books"] = df[book_cols].apply(lambda r: sum(_is_valid_num(x) for x in r.values), axis=1)

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        events = ["All"] + sorted(df["Event"].dropna().unique().tolist()) if "Event" in df.columns else ["All"]
        selected_event = st.selectbox("Event", events, index=0)
        bet_types = ["All"] + sorted(df["Bet Type"].dropna().unique().tolist()) if "Bet Type" in df.columns else ["All"]
        selected_bet_type = st.selectbox("Bet Type", bet_types, index=0)
        books = ["All"] + sorted(df["Best Book"].dropna().unique().tolist()) if "Best Book" in df.columns else ["All"]
        selected_book = st.selectbox("Best Book", books, index=0)
        min_books = st.number_input("Min. books posting this line", min_value=1, max_value=20, value=3, step=1)
        compact_mobile = st.toggle("Compact mobile mode (hide other books)", value=False)

    if selected_event != "All":
        df = df[df["Event"] == selected_event]
    if selected_bet_type != "All":
        df = df[df["Bet Type"].astype(str).str.strip() == selected_bet_type]
    if selected_book != "All":
        df = df[df["Best Book"] == selected_book]

    df = df[df["# Books"] >= int(min_books)]

    # Convert sportsbook columns
    for col in book_cols:
        df[col] = df[col].apply(to_american)
    if "Best Odds" in df.columns:
        df["Best Odds"] = df["Best Odds"].apply(to_american)

    # Convert ratio to percentage string (e.g. 2.904762 → 290.5%)
    df["% Over Market Avg"] = pd.to_numeric(df["% Over Market Avg"], errors="coerce")
    df["% Over Market Avg"] = df["% Over Market Avg"].apply(
        lambda x: f"{(x * 100):.1f}%" if pd.notnull(x) else ""
    )


    # --- Kelly & Z Calculations ---
    def _kelly(row, alpha=8, eps=0.6):
        try:
            p_fair = 1 / float(row["avg_other"])
            d = float(row["best_decimal"])
            b = d - 1
            if b <= 0 or p_fair <= 0 or p_fair >= 1:
                return float("nan")
            f_k = (b * p_fair - (1 - p_fair)) / b
            f_k = max(0, f_k)
            denom = max(math.log(1 + alpha * (d - 1)), eps)
            return f_k / denom
        except Exception:
            return float("nan")

    def _z_score(row, delta=1e-6):
        """Robust z-score of best book implied prob vs. market median."""
        try:
            # get best book implied prob
            d_best = float(row.get("best_decimal", float("nan")))
            if pd.isna(d_best) or d_best <= 1:
                return float("nan")
            p_best = 1.0 / d_best
    
            # collect all implied probabilities across books
            probs = []
            for c in book_cols:
                v = row.get(c)
                if pd.isna(v):
                    continue
                # handle both American (+120) and decimal (1.85) formats
                s = str(v).strip()
                if s == "":
                    continue
                if s.startswith("+") or s.startswith("-"):
                    try:
                        a = float(s)
                        if a > 0:
                            dec = 1 + (a / 100)
                        else:
                            dec = 1 + (100 / abs(a))
                        probs.append(1 / dec)
                    except Exception:
                        continue
                else:
                    try:
                        dec = float(s)
                        if dec > 1:
                            probs.append(1 / dec)
                    except Exception:
                        continue
    
            if len(probs) < 3:
                return float("nan")
    
            series = pd.Series(probs)
            m = series.median()
            mad = (series - m).abs().median() * 1.4826  # robust MAD scale factor
            return (m - p_best) / (mad + delta)
        except Exception:
            return float("nan")


    df["Kelly"] = df.apply(_kelly, axis=1)
    df["Z"] = df.apply(_z_score, axis=1)

    # Sort by Kelly
    df = df.sort_values(by=["Kelly"], ascending=False, na_position="last")

    # --- Columns for display ---
    base_cols = ["Event", "Player", "Bet Type", "Alt Line"]
    is_mobile = st.session_state.get("is_mobile", False)
    if selected_book != "All" and (is_mobile or compact_mobile):
        odds_cols_to_show = [selected_book] if selected_book in book_cols else []
    else:
        odds_cols_to_show = book_cols.copy()

    display_cols = (
        base_cols + odds_cols_to_show + ["% Over Market Avg", "Kelly", "Z"]
    )
    # hide internal columns
    hidden = {"# Books", "Best Book", "Best Odds"}
    display_cols = [c for c in display_cols if c in df.columns and c not in hidden]
    render_df = df[display_cols].copy()

    # --- Styling ---
    def step_style(val, step_size=0.1, floor=0.0, cap=4.0):
        try:
            v = float(val)
        except Exception:
            return ""
        if pd.isna(v) or v <= floor:
            return ""
        capped = min(v, cap)
        steps = int((capped - floor) // step_size)
        alpha = 0.12 + (0.95 - 0.12) * (steps / max(1, int((cap - floor) / step_size)))
        return f"background-color: rgba(34,139,34,{alpha}); font-weight: 600;"

    styled = render_df.style
    if "Kelly" in render_df.columns:
        styled = styled.applymap(step_style, subset=["Kelly"])
    if "Z" in render_df.columns:
        styled = styled.applymap(step_style, subset=["Z"])
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'),
                                     ('text-align', 'center'),
                                     ('font-size', '16px'),
                                     ('background-color', '#003366'),
                                     ('color', 'white')]}
    ])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

if __name__ == "__main__":
    run_app()
