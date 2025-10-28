import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import pytz
import requests
import time
import math
import streamlit.components.v1 as components  # used for a tiny width sniff (optional)

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

# --- (Very light client-width sniff; sets st.session_state["is_mobile"]) ---
# NOTE: This is deliberately minimal and safe. If it fails, we fall back to sidebar toggle.
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
# Streamlit can't directly read postMessage; we give the user a manual toggle fallback below.

# ---------- GitHub Actions Trigger (manual refresh) ----------
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

# ---------- Top controls ----------
btn_cols = st.columns([1, 1, 1])
with btn_cols[1]:
    if st.button("Refresh Odds", use_container_width=True):
        if trigger_github_action():
            wait_for_csv_update()

# ---------- Clear cache (helps some iPhone loads) ----------
with st.sidebar:
    if st.button("Clear Cache (fix stuck loads)"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        st.success("Cache cleared. Reloading…")
        st.experimental_rerun()

def run_app(df: pd.DataFrame | None = None):
    # ---- Load CSV ----
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

    # ---- Rename for display ----
    rename_map = {
        "event": "Event",
        "player": "Player",
        "group": "Bet Type",
        "threshold": "Alt Line",
        "best_book": "Best Book",
        "best_odds": "Best Odds",
        "value_ratio": "% Over Market Avg",  # rename per request
    }
    df = df.rename(columns=rename_map)

    if "Bet Type" in df.columns:
        df["Bet Type"] = df["Bet Type"].astype(str).str.strip()

    # ---- Identify sportsbook columns BEFORE formatting to strings ----
    fixed_cols_raw = {
        "Event", "Player", "Bet Type", "Alt Line", "Best Book", "Best Odds",
        "% Over Market Avg", "best_decimal", "avg_other"
    }
    all_cols = list(df.columns)
    book_cols = [c for c in all_cols if c not in fixed_cols_raw and not str(c).startswith("Unnamed")]

    # ---- Books available count (non-null numeric odds) ----
    def _is_valid_num(v):
        try:
            return pd.notnull(v) and str(v).strip() != ""
        except Exception:
            return False

    books_available = df[book_cols].apply(lambda r: sum(_is_valid_num(x) for x in r.values), axis=1)
    df["# Books"] = books_available

    # ---- Sidebar Filters ----
    with st.sidebar:
        st.header("Filters")
        events = ["All"] + sorted(df["Event"].dropna().unique().tolist()) if "Event" in df.columns else ["All"]
        selected_event = st.selectbox("Event", events, index=0)

        bet_types = ["All"] + sorted(df["Bet Type"].dropna().unique().tolist()) if "Bet Type" in df.columns else ["All"]
        selected_bet_type = st.selectbox("Bet Type", bet_types, index=0)

        books = ["All"] + sorted(df["Best Book"].dropna().unique().tolist()) if "Best Book" in df.columns else ["All"]
        selected_book = st.selectbox("Best Book", books, index=0)

        # NEW: min-books filter (default 3)
        min_books = st.number_input("Min. books posting this line", min_value=1, max_value=20, value=3, step=1)

        # Fallback toggle for compact mobile mode; ONLY affects phones, but user can force it if auto-detect fails
        compact_mobile = st.toggle("Compact mobile mode (hide other books)", value=False,
                                   help="On phones, hide all sportsbook columns except the selected Best Book.")

    # ---- Apply filters ----
    if selected_event != "All":
        df = df[df["Event"] == selected_event]
    if selected_bet_type != "All":
        df = df[df["Bet Type"].astype(str).str.strip() == selected_bet_type]
    if selected_book != "All":
        df = df[df["Best Book"] == selected_book]

    # Min-books filter
    df = df[df["# Books"] >= int(min_books)]

    # ---- Compute displays / formatting ----
    # Convert sportsbook columns to American (strings) for display
    for col in book_cols:
        df[col] = df[col].apply(to_american)
    if "Best Odds" in df.columns:
        df["Best Odds"] = df["Best Odds"].apply(to_american)

    # Ensure numeric for calc fields
    pct_over_ratio = pd.to_numeric(df.get("% Over Market Avg"), errors="coerce")
    best_decimal = pd.to_numeric(df.get("best_decimal"), errors="coerce")

    # Display % as percent (not ratio)
    df["% Over Market Avg (disp)"] = ((pct_over_ratio - 1.0) * 100.0).map(
        lambda x: f"{x:.1f}%" if pd.notnull(x) else ""
    )

    # NEW: Suggested Unit Size (log decay using decimal odds as multiplier)
    def _suggested_unit(row):
        val_ratio = row.get("% Over Market Avg")
        dec = row.get("best_decimal")
        try:
            v = float(val_ratio)
            d = float(dec)
            if pd.isna(v) or pd.isna(d) or d <= 1.0:
                return float("nan")
            # Use natural log for decay; higher odds (bigger d) => bigger denominator => smaller score
            return v / math.log(d)
        except Exception:
            return float("nan")

    df["Suggested Unit Size"] = df.apply(_suggested_unit, axis=1)

    # Sorting: by Suggested Unit Size (desc)
    df = df.sort_values(by=["Suggested Unit Size"], ascending=False, na_position="last")

    # ---- Build render dataframe ----
    # Columns to show by default (desktop)
    base_cols = ["Event", "Player", "Bet Type", "Alt Line"]

    # Mobile compact logic:
    # Only when device == mobile AND a Best Book is selected do we hide other books.
    # We expose a user toggle fallback; if not mobile, we keep desktop behavior.
    # (Auto-detect is best-effort; desktop is unaffected.)
    is_mobile = st.session_state.get("is_mobile", False)
    # If Best Book filter is selected AND mobile -> show only that book plus metrics
    if selected_book != "All" and (is_mobile or compact_mobile):
        odds_cols_to_show = [selected_book] if selected_book in book_cols else []
    else:
        odds_cols_to_show = book_cols.copy()

    # Final column plan
    display_cols = (
        base_cols
        + odds_cols_to_show
        + ["% Over Market Avg (disp)", "Suggested Unit Size", "Best Book", "Best Odds", "# Books"]
    )
    display_cols = [c for c in display_cols if c in df.columns]
    render_df = df[display_cols].copy()

    # ---- Styling ----
    # We now style only "Suggested Unit Size" (step size 0.1). The old % column is plain text.
    def step_style(val, step_size=0.1, floor=1.0, cap=4.0):
        try:
            v = float(val)
        except Exception:
            return ""
        if pd.isna(v) or v <= floor:
            return ""
        capped = min(v, cap)
        steps = int((capped - floor) // step_size)
        steps = max(0, min(steps, int((cap - floor) / step_size)))
        alpha = 0.12 + (0.95 - 0.12) * (steps / max(1, int((cap - floor) / step_size)))
        return f"background-color: rgba(34,139,34,{alpha}); font-weight: 600;"

    def highlight_best_book_cells(row):
        styles = [""] * len(row)
        best = row.get("Best Book", "")
        shade = step_style(row.get("Suggested Unit Size", ""))
        if best and best in row.index:
            idx = list(row.index).index(best)
            styles[idx] = shade
        return styles

    styled = render_df.style
    if "Suggested Unit Size" in render_df.columns:
        styled = styled.applymap(step_style, subset=["Suggested Unit Size"])
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
