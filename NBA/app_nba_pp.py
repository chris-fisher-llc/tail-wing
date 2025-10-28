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

# --- Auto-clear cache each run (helps mobile stuck sessions) ---
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

# ---------- One-time mobile flag bootstrap (hard redirect; no autorefresh) ----------
qp = st.query_params
if "mobile" not in qp:
    components.html(
        """
        <script>
          (function() {
            const w = Math.min(window.innerWidth || 9999, screen.width || 9999);
            const isMobile = w < 800;
            try {
              const url = new URL(window.location);
              url.searchParams.set('mobile', isMobile ? '1' : '0');
              // Hard reload so Python sees the param immediately
              window.location.replace(url.toString());
            } catch(e) {}
          })();
        </script>
        """,
        height=0,
    )
    st.write("Loading…")
    st.stop()

mobile_flag = st.query_params.get("mobile")
if isinstance(mobile_flag, list):
    mobile_flag = mobile_flag[0] if mobile_flag else None
auto_mobile = (mobile_flag == "1")

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

    # --- Rename and setup
    rename_map = {
        "event": "Event",
        "player": "Player",
        "group": "Bet Type",
        "threshold": "Alt Line",
        "best_book": "Best Book",
        "best_odds": "Best Odds",
        "value_ratio": "Line vs. Average",  # ratio
    }
    df = df.rename(columns=rename_map)

    # Identify sportsbook columns (raw before we format display)
    fixed_cols_raw = {
        "Event", "Player", "Bet Type", "Alt Line", "Best Book", "Best Odds",
        "Line vs. Average", "best_decimal", "avg_other"
    }
    all_cols = list(df.columns)
    book_cols = [c for c in all_cols if c not in fixed_cols_raw and not str(c).startswith("Unnamed")]

    # Count books posting
    def _is_valid_num(v):
        try:
            return pd.notnull(v) and str(v).strip() != ""
        except Exception:
            return False
    df["# Books"] = df[book_cols].apply(lambda r: sum(_is_valid_num(x) for x in r.values), axis=1)

    # --- Sidebar filters
    with st.sidebar:
        st.header("Filters")
        events = ["All"] + sorted(df["Event"].dropna().unique().tolist()) if "Event" in df.columns else ["All"]
        selected_event = st.selectbox("Event", events, index=0)

        bet_types = ["All"] + sorted(df["Bet Type"].dropna().unique().tolist()) if "Bet Type" in df.columns else ["All"]
        selected_bet_type = st.selectbox("Bet Type", bet_types, index=0)

        books = ["All"] + sorted(df["Best Book"].dropna().unique().tolist()) if "Best Book" in df.columns else ["All"]
        selected_book = st.selectbox("Best Book", books, index=0)

        min_books = st.number_input("Min. books posting this line", min_value=1, max_value=20, value=4, step=1)

        # DEFAULT ON when mobile=1
        compact_mobile = st.toggle("Compact mobile mode (hide other books)", value=bool(auto_mobile))

    # Apply filters
    if selected_event != "All":
        df = df[df["Event"] == selected_event]
    if selected_bet_type != "All":
        df = df[df["Bet Type"].astype(str).str.strip() == selected_bet_type]
    if selected_book != "All":
        df = df[df["Best Book"] == selected_book]

    df = df[df["# Books"] >= int(min_books)]

    # Convert sportsbook columns (for display)
    for col in book_cols:
        df[col] = df[col].apply(to_american)
    if "Best Odds" in df.columns:
        df["Best Odds"] = df["Best Odds"].apply(to_american)

    # --- Numeric percent for Line vs. Average (ratio -> %)
    df["Line vs. Average"] = pd.to_numeric(df["Line vs. Average"], errors="coerce")
    df["Line vs. Average (%)"] = (df["Line vs. Average"] - 1.0) * 100.0

    # --- Significance score (unchanged core logic)
    def _significance(row, alpha=8.0, eps=0.6):
        try:
            ratio = float(row.get("Line vs. Average", float("nan")))
            d = float(row.get("best_decimal", float("nan")))
            n = int(row.get("# Books", 0))
            if not (ratio > 0 and d > 1):
                return float("nan")
            edge_pct = (ratio - 1.0) * 100.0
            denom = max(math.log(1.0 + alpha * (d - 1.0)), eps)
            quality = min(1.0, max(0.0, (n - 3) / 3.0))
            score = (edge_pct * quality) / denom
            return max(-50.0, min(50.0, score))
        except Exception:
            return float("nan")

    df["Significance"] = df.apply(_significance, axis=1)

    # --- Build render dataframe (pre-sorted numerically) ---
    base_cols = ["Event", "Player", "Bet Type", "Alt Line"]

    # Effective mobile mode
    is_mobile = bool(auto_mobile or compact_mobile)

    # Decide which odds columns to show
    if selected_book != "All" and is_mobile:
        odds_cols_to_show = [selected_book] if selected_book in book_cols else []
    else:
        odds_cols_to_show = book_cols.copy()

    hidden = {"# Books", "Best Book", "Best Odds"}

    display_cols = base_cols + odds_cols_to_show + ["Line vs. Average (%)", "Significance"]
    display_cols = [c for c in display_cols if c in df.columns and c not in hidden]

    render_df = df[display_cols].copy()
    render_df = render_df.sort_values(by=["Significance"], ascending=False, na_position="last")

    # --- Conditional green shading by Significance (0.25 steps) ---
    def _sig_green(val):
        try:
            s = float(val)
        except Exception:
            return ""
        if s <= 0:
            return ""
        cap = 10.0  # UI sanity cap
        step_size = 0.25
        max_steps = int(cap / step_size)  # 40 steps
        steps = max(0, min(int(s // step_size), max_steps))
        alpha = 0.12 + (0.95 - 0.12) * (steps / max_steps)
        return f"background-color: rgba(34,139,34,{alpha}); font-weight: 600;"

    def _shade_selected_book(row, target_col: str):
        styles = [""] * len(row)
        if target_col and target_col in row.index:
            shade = _sig_green(row.get("Significance", 0))
            try:
                idx = list(row.index).index(target_col)
                styles[idx] = shade
            except Exception:
                pass
        return styles

    # --- Mobile table cosmetics
    if is_mobile:
        st.markdown(
            """
            <style>
              [data-testid="stDataFrame"] * { font-size: 0.92rem !important; }
            </style>
            """,
            unsafe_allow_html=True
        )

    # --- Build Styler with formatting + shading ---
    styled = render_df.style

    # Shade Significance column
    if "Significance" in render_df.columns:
        styled = styled.applymap(_sig_green, subset=["Significance"])

    # Shade selected book column (if any and visible)
    target_col = selected_book if (selected_book != "All" and selected_book in render_df.columns) else None
    if target_col:
        styled = styled.apply(_shade_selected_book, axis=1, target_col=target_col)

    # Header style + numeric formats
    styled = (
        styled.format({
            "Line vs. Average (%)": "{:.1f}%",
            "Significance": "{:.2f}",
        })
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('font-size', '16px'),
                ('background-color', '#003366'),
                ('color', 'white')
            ]}
        ])
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

if __name__ == "__main__":
    run_app()
