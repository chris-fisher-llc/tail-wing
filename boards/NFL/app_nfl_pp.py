# app_nfl_pp.py — The Tail Wing (NFL Player Props)
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import pytz
import requests
from typing import Optional

# -------------------------
# Page setup + light CSS
# -------------------------
st.set_page_config(page_title="The Tail Wing - NFL Player Props", layout="wide")
st.markdown("""
<style>
/* Tighten vertical rhythm */
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; }

/* CTA "cards" */
.cta-card {
  border: 1px solid #e0e3e8; border-radius: 14px; padding: 16px 18px; background: #f9fbff;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.cta-card h3 { margin: 0 0 6px 0; font-size: 20px; }
.cta-muted { color: #5f6b7a; font-size: 13px; margin-top: 6px; }

/* Section headers */
h2.section { margin: 14px 0 8px 0; font-size: 22px; }

/* Align CTA rows */
.cta-row { display: flex; gap: 16px; align-items: center; }
.cta-row .item { flex: 1; }

/* Make link buttons feel primary */
a[role="button"] {
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Secrets / Config
# -------------------------
PAYWALL_API  = st.secrets["PAYWALL_API"]           # e.g., https://high-lines-backend.onrender.com
FRONTEND_URL = st.secrets["FRONTEND_URL"]          # e.g., https://high-lines-nfl-players.streamlit.app
PRICE_ALL    = st.secrets.get("STRIPE_PRICE_ALL", "")
PRICE_SINGLE = st.secrets.get("STRIPE_PRICE_SINGLE", PRICE_ALL)  # (same plan for now)

# Optional GitHub refresh
GITHUB_TOKEN         = st.secrets.get("GITHUB_TOKEN")
GITHUB_REPO          = st.secrets.get("GITHUB_REPO")  # "owner/repo"
GITHUB_WORKFLOW_FILE = st.secrets.get("GITHUB_WORKFLOW_FILE", "update-nfl-player-props.yml")
GITHUB_REF           = st.secrets.get("GITHUB_REF", "main")

# -------------------------
# Stripe return (new API)
# -------------------------
qp = st.query_params
if qp.get("status") == "success":
    st.success("Subscription activated. Loading full board…")
    st.query_params.clear()

# -------------------------
# Auth helpers
# -------------------------
def _auth_headers():
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}

def _sign_out():
    st.session_state.pop("token", None)
    st.session_state.pop("email", None)
    st.rerun()

def _entitlement_soft():
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
        return "signed_out", False, None

def _dev_login_form():
    st.markdown('<div class="cta-card">', unsafe_allow_html=True)
    st.markdown("### Sign in", unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False, border=False):
        email = st.text_input("Email", placeholder="you@example.com")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            try:
                r = requests.post(f"{PAYWALL_API}/auth/dev_login", json={"email": email}, timeout=10)
                r.raise_for_status()
                st.session_state["token"] = r.json()["session_token"]
                st.session_state["email"] = email
                st.rerun()
            except Exception as e:
                st.error("No account found or sign-in failed.")
    st.markdown('<div class="cta-muted">Existing subscribers: sign in to unlock the full board.</div></div>', unsafe_allow_html=True)

def _checkout_url(email: str) -> str:
    """Ask backend for a Stripe Checkout URL. Backend sets success/cancel URLs."""
    payload = {"email": email, "referrer": None}
    r = requests.post(f"{PAYWALL_API}/billing/checkout", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()["url"]

# -------------------------
# Data loading
# -------------------------
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

def _load_df() -> Optional[pd.DataFrame]:
    csv_path = _find_csv_path()
    if not csv_path or not csv_path.exists():
        st.error("nfl_player_props.csv not found. Place it next to this app or in an 'nfl/' subfolder.")
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
        st.error(f"Error loading CSV: {e}")
        return None

# -------------------------
# Header
# -------------------------
st.markdown(
    """
    <h1 style='text-align: center; font-size: 42px; margin-bottom:0;'>
       🏈 NFL Player Props — Anomaly Board 🏈
    </h1>
    <p style='text-align: center; font-size:18px; color: gray; margin-top:6px;'>
        Powered by The Tail Wing — scanning books for alt-yardage & anytime TD edges
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Refresh Odds (centered)
# -------------------------
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
        st.success("Refresh kicked off. Odds will update on CSV push.")
    else:
        st.error(f"Failed to trigger workflow ({resp.status_code}): {resp.text}")

mid = st.columns([1,1,1])[1]
with mid:
    st.link_button("Refresh Odds", "#", use_container_width=True)  # visual emphasis
    # (If you want functional refresh, swap link_button for st.button + trigger call)
    # if st.button("Refresh Odds", use_container_width=True): trigger_github_action()

# -------------------------
# Load data & normalize
# -------------------------
df = _load_df()
if df is None or df.empty:
    st.stop()

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

# -------------------------
# Sidebar: Account + filters
# -------------------------
with st.sidebar:
    st.caption("Account")
    if st.session_state.get("token"):
        st.write(st.session_state.get("email", ""))
        if st.button("Sign out", use_container_width=True):
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

def value_step_style(val):
    try:
        v = float(val)
    except Exception:
        return ""
    if v <= 1.0: return ""
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

# -------------------------
# Paywall + Render
# -------------------------
def render_board():
    status, is_sub, email = _entitlement_soft()

    # Full board if subscribed
    if status == "signed_in" and is_sub:
        st.success("Subscriber view: full board unlocked.")
        st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)
        return

    # Preview
    st.markdown("### Current Snapshot")
    st.write("**Free preview** — top edges with banded EV. Updated every ~10 minutes.")
    st.table(_to_teaser(render_df))
    st.info("Subscribe to unlock exact EV%, best book & price, historical movement, and CSV export.")

    # CTA row: Sign-in card + Subscribe card (aligned + styled)
    st.markdown('<div class="cta-row">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="cta-card">', unsafe_allow_html=True)
        st.markdown("<h3>Sign in (existing subscribers)</h3>", unsafe_allow_html=True)
        if status == "signed_out" and not st.session_state.get("show_login", False):
            if st.button("Sign in", use_container_width=True):
                st.session_state["show_login"] = True
                st.rerun()
        if st.session_state.get("show_login", False):
            _dev_login_form()
        if status == "signed_in" and not is_sub:
            st.write(f"Signed in as {email or ''}")
            st.caption("Account found, but no active subscription.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="cta-card">', unsafe_allow_html=True)
        st.markdown("<h3>Subscribe</h3>", unsafe_allow_html=True)
        if not st.session_state.get("show_plans"):
            if st.button("Choose a plan", type="primary", use_container_width=True):
                st.session_state["show_plans"] = True
                st.rerun()
        else:
            # plan buttons side-by-side
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Single Board – $5**")
                if st.button("Select", use_container_width=True):
                    st.session_state["selected_plan"] = "single"
                    st.session_state["show_email_for_checkout"] = True
                    st.rerun()
            with p2:
                st.markdown("**All Boards – $9**")
                if st.button("Select", use_container_width=True):
                    st.session_state["selected_plan"] = "all"
                    st.session_state["show_email_for_checkout"] = True
                    st.rerun()

            # Email capture (fixes 422) — works signed-in or signed-out
            if st.session_state.get("show_email_for_checkout"):
                st.markdown("—")
                default_email = st.session_state.get("email", "")
                email_in = st.text_input("Email for checkout", value=default_email, placeholder="you@example.com", key="checkout_email")
                if st.link_button("Proceed to Stripe Checkout →",
                                  "#",  # visual only; we show a real link below when ready
                                  use_container_width=True):
                    pass  # link_button consumes click; we gate on the real button below
                if st.button("Get secure checkout link", type="primary", use_container_width=True):
                    if not email_in:
                        st.error("Please enter an email to continue.")
                    else:
                        try:
                            url = _checkout_url(email_in)
                            st.link_button("Open Stripe Checkout", url, type="primary", use_container_width=True)
                            st.caption("If the button doesn’t open, copy the URL into a new tab.")
                        except Exception as e:
                            st.error(f"Checkout failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

render_board()
