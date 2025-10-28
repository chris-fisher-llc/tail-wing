# app_nfl_pp.py — The Tail Wing (NFL Player Props)
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import pytz
import requests
from typing import Optional

# ---- PAGE SETUP ----
st.set_page_config(page_title="The Tail Wing - NFL Player Props", layout="wide")

# ---- Config (from Streamlit Secrets) ----
PAYWALL_API  = st.secrets["PAYWALL_API"]           # e.g., https://high-lines-backend.onrender.com
FRONTEND_URL = st.secrets["FRONTEND_URL"]          # e.g., https://high-lines-nfl-players.streamlit.app
PRICE_ALL    = st.secrets.get("STRIPE_PRICE_ALL", "")
PRICE_SINGLE = st.secrets.get("STRIPE_PRICE_SINGLE", PRICE_ALL)  # (same plan for now)

# Optional GitHub refresh hook
GITHUB_TOKEN         = st.secrets.get("GITHUB_TOKEN")
GITHUB_REPO          = st.secrets.get("GITHUB_REPO")  # "owner/repo"
GITHUB_WORKFLOW_FILE = st.secrets.get("GITHUB_WORKFLOW_FILE", "update-nfl-player-props.yml")
GITHUB_REF           = st.secrets.get("GITHUB_REF", "main")

# ---- Handle return from Stripe Checkout (new API) ----
qp = st.query_params  # property, not callable
if qp.get("status") == "success":
    st.success("Subscription activated. Loading full board…")
    st.query_params.clear()  # clean URL

# ---- Auth helpers ----
def _auth_headers():
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}

def _sign_out():
    st.session_state.pop("token", None)
    st.session_state.pop("email", None)
    st.rerun()

def _entitlement_soft():
    """
    Soft check: if token present, query /entitlement.
    Returns (status, is_subscriber, email) where status in {'signed_out','signed_in'}.
    """
    token = st.session_state.get("token")
    if not token:
        return "signed_out", False, None
    try:
        r = requests.get(f"{PAYWALL_API}/entitlement", headers=_auth_headers(), timeout=10)
        if r.status_code == 401:
            _sign_out()
        r.raise_for_status()
        js = r.json()
        return "signed_in", bool(js.get("is_subscriber")), js.get("email")
    except Exception:
        # Fail safe to preview
        return "signed_out", False, None

def _dev_login_ui():
    with st.form("login_form", clear_on_submit=False, border=True):
        email = st.text_input("Email", placeholder="you@example.com")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            r = requests.post(f"{PAYWALL_API}/auth/dev_login", json={"email": email}, timeout=10)
            r.raise_for_status()
            st.session_state["token"] = r.json()["session_token"]
            st.session_state["email"] = email
            st.rerun()

def _checkout_url() -> str:
    """
    Get a Stripe Checkout URL from backend. Backend builds success/cancel URLs.
    Requires st.session_state['email'] to be set (prompt sign-in if missing).
    """
    payload = {"email": st.session_state.get("email"), "referrer": None}
    r = requests.post(f"{PAYWALL_API}/billing/checkout", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()["url"]

# ---- CSV path resolution ----
def _find_csv_path() -> Optional[Path]:
    env = os.getenv("NFL_PROPS_CSV")
    if env:
        p = Path(env)
        if p.exists():
            return p

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

    try:
        for p in here.rglob("nfl_player_props.csv"):
            return p
    except Exception:
        pass
    return None

# ---- Manual refresh (GitHub Actions trigger) ----
def trigger_github_action():
    if not (GITHUB_TOKEN and GITHUB_REPO):
        st.error("Missing secrets: set GITHUB_TOKEN and GITHUB_REPO to enable refresh.")
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW_FILE}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"ref": GITHUB_REF}

    with st.spinner("Triggering GitHub Action…"):
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
    if resp.status_code == 204:
        st.success("Refresh kicked off. Odds will update automatically when the CSV is pushed.")
    else:
        st.error(f"Failed to trigger workflow ({resp.status_code}): {resp.text}")

def wait_for_csv_update():
    import time
    csv_path = _find_csv_path()
    if csv_path and csv_path.exists():
        old_mtime = csv_path.stat().st_mtime
        for _ in range(12):  # check every 10s for 2 minutes
            time.sleep(10)
            if csv_path.stat().st_mtime != old_mtime:
                st.success("Data updated — reloading!")
                st.rerun()

# ---- Header ----
st.markdown(
    """
    <h1 style='text-align: center; font-size: 42px;'>
       🏈 NFL Player Props — Anomaly Board 🏈
    </h1>
    <p style='text-align: center; font-size:18px; color: gray;'>
        Powered by The Tail Wing — scanning books for alt-yardage & anytime TD edges
    </p>
    """,
    unsafe_allow_html=True
)

# ---- Refresh button (centered) ----
btn_cols = st.columns([1, 1, 1])
with btn_cols[1]:
    if st.button("Refresh Odds", use_container_width=True):
        trigger_github_action()
        st.info("Waiting for new data...")
        wait_for_csv_update()

# ---- Load CSV ----
def _load_df() -> Optional[pd.DataFrame]:
    csv_path = _find_csv_path()
    if not csv_path or not csv_path.exists():
        st.error(
            "nfl_player_props.csv not found.\n\n"
            "• Place the file next to this app, or in an 'nfl/' subfolder, or set env var NFL_PROPS_CSV to the full path.\n\n"
            f"Working directory: {Path.cwd()}\n"
        )
        return None
    try:
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed')]
        df = df.dropna(axis=1, how="all")
        to_zone = pytz.timezone('US/Eastern')
        ts = datetime.fromtimestamp(csv_path.stat().st_mtime, pytz.utc)
        eastern = ts.astimezone(to_zone).strftime("%Y-%m-%d %I:%M %p %Z")
        st.caption(f"Odds last updated: {eastern}")
        return df
    except Exception as e:
        st.error(f"Error loading {csv_path}: {e}")
        return None

df = _load_df()
if df is None:
    st.stop()
if df.empty:
    st.warning("No data to display.")
    st.stop()

# ---- Normalize/format ----
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

def _american(x):
    try:
        x = int(float(x))
        return f"+{x}" if x > 0 else str(x)
    except Exception:
        return ""

for col in odds_cols:
    df[col] = df[col].apply(_american)
if "Best Odds" in df.columns:
    df["Best Odds"] = df["Best Odds"].apply(_american)

df["Value"] = pd.to_numeric(df.get("Value"), errors="coerce")
df["_Value_print"] = df["Value"].map(lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if pd.notnull(x) else "")

display_cols = ["Event", "Player", "Bet Type", "Alt Line"] + odds_cols + ["Value", "_Value_print", "Best Book", "Best Odds"]
display_cols = [c for c in display_cols if c in df.columns]
df = df[display_cols].copy()

# ---- Sidebar: Account + Filters ----
with st.sidebar:
    st.caption("Account")
    if st.session_state.get("token"):
        st.write(st.session_state.get("email", ""))
        if st.button("Sign out"):
            _sign_out()
    else:
        st.caption("You’re viewing the free preview.")

with st.sidebar:
    st.header("Best Book")
    books = df["Best Book"].dropna().unique().tolist() if "Best Book" in df.columns else []
    selected_book = st.selectbox("", ["All"] + sorted(books))

    st.header("Event")
    events = df["Event"].dropna().unique().tolist() if "Event" in df.columns else []
    selected_event = st.selectbox("", ["All"] + sorted(events))

    st.header("Bet Type")
    bet_types = sorted(df["Bet Type"].dropna().unique().tolist()) if "Bet Type" in df.columns else []
    selected_bet_type = st.selectbox("", ["All"] + bet_types)

# Apply filters
if selected_book != "All":
    df = df[df["Best Book"] == selected_book]
if selected_event != "All":
    df = df[df["Event"] == selected_event]
if selected_bet_type != "All":
    df = df[df["Bet Type"].astype(str).str.strip() == selected_bet_type]

if "Value" in df.columns:
    df = df.sort_values(by="Value", ascending=False, na_position="last")

df["Value_display"] = df["_Value_print"]
df.drop(columns=["_Value_print"], inplace=True)

# ---- Styling ----
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

# ---- Teaser utility ----
def _to_teaser(df_full: pd.DataFrame) -> pd.DataFrame:
    safe_cols = [c for c in df_full.columns if c.lower() not in {"ev%", "ev", "best book", "best_book", "best price", "best_price", "price"}]
    out = df_full[safe_cols].copy()
    ev_col = next((c for c in df_full.columns if c.lower() in {"ev%", "ev", "value"}), None)
    if ev_col:
        def band(x):
            try: v = float(x)
            except Exception: return "—"
            if v < 2: return "<2%"
            if v < 4: return "2–4%"
            if v < 6: return "4–6%"
            if v < 9: return "6–9%"
            return "9%+"
        out["Edge Band"] = pd.to_numeric(df_full[ev_col], errors="coerce").map(band)
    return out.head(3)

# ---- Paywall + Render ----
def render_board():
    status, is_sub, email = _entitlement_soft()

    # Subscriber → full board
    if status == "signed_in" and is_sub:
        st.success("Subscriber view: full board unlocked.")
        st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)
        return

    # Preview always visible
    st.markdown("### Current Snapshot")
    st.write("**Free preview** — top edges with banded EV. Updated every ~10 minutes.")
    st.table(_to_teaser(render_df))
    st.info("Subscribe to unlock exact EV%, best book & price, historical movement, and CSV export.")

    # Actions
    c1, c2 = st.columns(2)
    with c1:
        if status == "signed_out":
            if st.button("Sign in (existing subscribers)"):
                _dev_login_ui()
                st.stop()
        else:
            st.write(f"Signed in as {email or ''}")
            st.caption("Not subscribed.")
    with c2:
        if st.button("Subscribe"):
            st.session_state["show_plans"] = True

    # Plans (both map to same backend price for now)
    if st.session_state.get("show_plans"):
        p1, p2 = st.columns(2)
        with p1:
            if st.button("Single Board Subscription ($5)"):
                try:
                    url = _checkout_url()
                    st.link_button("Proceed to Stripe Checkout →", url, type="primary")
                    st.stop()
                except Exception as e:
                    st.error(f"Checkout failed: {e}")
        with p2:
            if st.button("All-Boards Subscription ($9)"):
                try:
                    url = _checkout_url()
                    st.link_button("Proceed to Stripe Checkout →", url, type="primary")
                    st.stop()
                except Exception as e:
                    st.error(f"Checkout failed: {e}")

    st.stop()

render_board()
