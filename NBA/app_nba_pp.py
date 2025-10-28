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

# --- Best-effort mobile hint (used only to toggle compact styling) ---
components.html(
    """
    <script>
      const w = Math.min(window.innerWidth || 9999, screen.width || 9999);
      const isMobile = w < 800;
      // We can't read postMessage directly in Python; expose a URL hash we can read if needed.
      if (isMobile && !location.hash.includes("mobile=1")) {
        try { history.replaceState({}, "", location.pathname + location.search + "#mobile=1"); } catch(e) {}
      }
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

    # --- Rename and setup
    rename_map = {
        "event": "Event",
        "player": "Player",
        "group": "Bet Type",
        "threshold": "Alt Line",
        "best_book": "Best Book",
        "best_odds": "Best Odds",
        "value_ratio": "Line vs. Average",  # rename the ratio column
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

        # Default now 4
        min_books = st.number_input("Min. books posting this line", min_value=1, max_value=20, value=4, step=1)

        # Fallback toggle for compact/mobile rendering
        compact_mobile = st.toggle("Compact mobile mode (hide other books)", value=False)

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

    # --- Proper numeric + display percent for "Line vs. Average"
    # Line vs. Average is a RATIO (e.g., 1.25). Create numeric percent + render as %.
    df["Line vs. Average"] = pd.to_numeric(df["Line vs. Average"], errors="coerce")          # ratio
    df["Line vs. Average (%)]"] = (df["Line vs. Average"] - 1.0) * 100.0                     # numeric percent

    # --- Simple, robust Significance score ---
    # Intuition: edge (%) tempered by payout multiple and market thinness
    #   edge_pct = (ratio-1)*100
    #   denom = ln(1 + alpha*(decimal-1))  with floors to avoid explosions for heavy favorites
    #   quality = min(1, max(0, (#books-3)/3))   (0 when 3 or fewer; ~1 once 6+ books)
    #   significance = (edge_pct * quality) / denom
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
            # Optional clamp to keep UI sane
            return max(-50.0, min(50.0, score))
        except Exception:
            return float("nan")

    df["Significance"] = df.apply(_significance, axis=1)

    # --- Build render dataframe (plain DataFrame so sorting is numeric) ---
    base_cols = ["Event", "Player", "Bet Type", "Alt Line"]

    # Best-book compactness on mobile (only keep the chosen book on small screens)
    is_mobile = ("#mobile=1" in st.experimental_get_query_params().get("", [""])) or compact_mobile
    if selected_book != "All" and (is_mobile or compact_mobile):
        odds_cols_to_show = [selected_book] if selected_book in book_cols else []
    else:
        odds_cols_to_show = book_cols.copy()

    # Hide internal columns from display
    hidden = {"# Books", "Best Book", "Best Odds"}

    display_cols = base_cols + odds_cols_to_show + ["Line vs. Average (%)]", "Significance"]
    display_cols = [c for c in display_cols if c in df.columns and c not in hidden]
    render_df = df[display_cols].copy()

    # Sort by Significance by default (numeric)
    render_df = render_df.sort_values(by=["Significance"], ascending=False, na_position="last")

    # --- Mobile-only table cosmetics (font size down; fix Event/Player widths) ---
    if is_mobile:
        st.markdown(
            """
            <style>
              /* Nudge overall table font-size on mobile */
              [data-testid="stDataFrame"] * { font-size: 0.92rem !important; }
            </style>
            """,
            unsafe_allow_html=True
        )

    # --- Render with numeric sorting + nice formatting ---
    # Use column_config so "Line vs. Average (%)" displays as percent but sorts numerically
    col_cfg = {
        "Event": st.column_config.TextColumn("Event", width="small" if is_mobile else "medium"),
        "Player": st.column_config.TextColumn("Player", width="small" if is_mobile else "medium"),
        "Line vs. Average (%)]": st.column_config.NumberColumn("Line vs. Average", format="%.1f%%"),
        "Significance": st.column_config.NumberColumn("Significance", help="Edge adjusted for payout and market depth", format="%.2f"),
    }

    st.dataframe(
        render_df,
        use_container_width=True,
        hide_index=True,
        height=1200,
        column_config=col_cfg,
    )

if __name__ == "__main__":
    run_app()
