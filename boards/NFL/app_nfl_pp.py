# app_nfl_pp.py — The Tail Wing (NFL Player Props) — polished UX + your st.secrets + correct gating

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import pytz
import requests
from typing import Optional, Tuple

# -------------------------
# Page + CSS (hierarchy, spacing)
# -------------------------
st.set_page_config(page_title="The Tail Wing - NFL Player Props", layout="wide")
st.markdown("""
<style>
/* Vertical rhythm */
.block-container { padding-top: 1.0rem; padding-bottom: 2.5rem; }

/* Title / subtitles */
.tw-title h1 {
  font-family: 'Poppins', sans-serif; font-weight: 700; letter-spacing: .2px; margin-bottom: .35rem;
}
.tw-subtitle { font-family: 'Montserrat', sans-serif; font-size: .95rem; color: #7a7f87; margin-bottom: 1.25rem; }

/* Section headers */
.tw-section h3, .tw-section h4 {
  font-family: 'Montserrat', sans-serif; font-weight: 600; margin-bottom: .5rem;
}

/* Cards */
.tw-card { border: 1px solid #edf0f2; border-radius: 16px; padding: 18px; background: #fff; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.tw-plan  { border: 1px solid #e7ebef; border-radius: 14px; padding: 14px; background: #fcfdff; }

/* Status banners */
.tw-banner { border-radius: 12px; padding: 10px 12px; font-weight: 600; margin-bottom: 10px; }
.tw-ok   { background:#ecfdf3; color:#027a48; border:1px solid #abefc6; }
.tw-warn { background:#fff7ed; color:#b45309; border:1px solid #fed7aa; }
.tw-err  { background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; }

/* Buttons (natural width) */
.stButton>button { border-radius: 12px; padding: 8px 14px; font-weight: 600; }

/* Narrow the email input visually by constraining its parent container */
.narrow { max-width: 420px; }

/* Table header */
thead tr th { font-weight: 700 !important; text-align: center !important; font-size: 16px !important;
  background-color: #003366 !important; color: #fff !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Secrets / Config (unchanged: st.secrets)
# -------------------------
PAYWALL_API  = st.secrets["PAYWALL_API"]           # e.g., https://your-backend
FRONTEND_URL = st.secrets["FRONTEND_URL"]          # e.g., https://high-lines-nfl-players.streamlit.app
PRICE_ALL    = st.secrets.get("STRIPE_PRICE_ALL", "")
PRICE_SINGLE = st.secrets.get("STRIPE_PRICE_SINGLE", PRICE_ALL)

# Optional GitHub refresh (kept from your original)
GITHUB_TOKEN         = st.secrets.get("GITHUB_TOKEN")
GITHUB_REPO          = st.secrets.get("GITHUB_REPO")  # "owner/repo"
GITHUB_WORKFLOW_FILE = st.secrets.get("GITHUB_WORKFLOW_FILE", "update-nfl-player-props.yml")
GITHUB_REF           = st.secrets.get("GITHUB_REF", "main")

# -------------------------
# Stripe return (kept)
# -------------------------
qp = st.query_params
if qp.get("status") == "success":
    st.success("Subscription activated. Loading full board…")
    st.query_params.clear()

# -------------------------
# Auth helpers (kept + small hardening)
# -------------------------
def _auth_headers():
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}

def _sign_out():
    st.session_state.pop("token", None)
    st.session_state.pop("email", None)
    st.rerun()

def _entitlement_soft() -> Tuple[str, bool, Optional[str]]:
    """
    Returns (status, is_subscriber, email)
    status ∈ {"signed_out", "signed_in"}
    """
    token = st.session_state.get("token")
    if not token:
        return "signed_out", False, None
    try:
        # your original endpoint
        r = requests.get(f"{PAYWALL_API}/entitlement", headers=_auth_headers(), timeout=10)
        if r.status_code == 401:
            _sign_out()
        r.raise_for_status()
        js = r.json()
        return "signed_in", bool(js.get("is_subscriber")), js.get("email")
    except Exception:
        return "signed_out", False, None

def _dev_login(email: str) -> bool:
    try:
        r = requests.post(f"{PAYWALL_API}/auth/dev_login", json={"email": email}, timeout=10)
        r.raise_for_status()
        st.session_state["token"] = r.json()["session_token"]
        st.session_state["email"] = email
        return True
    except Exception:
        return False

def _checkout_url(email: str) -> str:
    """Ask backend for a Stripe Checkout URL. Backend decides plan; your original route."""
    payload = {"email": email, "referrer": None}
    r = requests.post(f"{PAYWALL_API}/billing/checkout", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()["url"]

# -------------------------
# Data loading (original)
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
# Title + Refresh
# -------------------------
st.markdown('<div class="tw-title">', unsafe_allow_html=True)
st.title("NFL Player Props — Anomaly Board 🏈")
st.markdown('<div class="tw-subtitle">Powered by The Tail Wing — scanning books for alt-yardage & anytime TD edges</div></div>', unsafe_allow_html=True)

def _trigger_github_action():
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

top_mid = st.columns([1,1,1])[1]
with top_mid:
    if st.button("Refresh Odds"):
        _trigger_github_action()

# -------------------------
# Load + prep dataframe (your logic, kept)
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
render_df["Value"] = render_df["_Value_print"]
render_df.drop(columns=["_Value_print"], inplace=True)

styled = render_df.style
if "Value" in render_df.columns:
    styled = styled.applymap(value_step_style, subset=["Value"])
styled = styled.apply(highlight_best_book_cells, axis=1)

# -------------------------
# Sidebar: Account + filters (kept; includes Sign out)
# -------------------------
with st.sidebar:
    st.caption("Account")
    if st.session_state.get("token"):
        st.write(st.session_state.get("email", ""))
        if st.button("Sign out", use_container_width=True, key="sidebar_signout"):
            _sign_out()
    else:
        st.caption("You’re viewing the free preview.")

with st.sidebar:
    st.header("Best Book")
    books = render_df["Best Book"].dropna().unique().tolist() if "Best Book" in render_df.columns else []
    selected_book = st.selectbox("", ["All"] + sorted(books))
    st.header("Event")
    events = render_df["Event"].dropna().unique().tolist() if "Event" in render_df.columns else []
    selected_event = st.selectbox("", ["All"] + sorted(events))
    st.header("Bet Type")
    bet_types = sorted(render_df["Bet Type"].dropna().unique().tolist()) if "Bet Type" in render_df.columns else []
    selected_bet_type = st.selectbox("", ["All"] + bet_types)

# Apply filters
df_filtered = render_df.copy()
if selected_book != "All":
    df_filtered = df_filtered[df_filtered["Best Book"] == selected_book]
if selected_event != "All":
    df_filtered = df_filtered[df_filtered["Event"] == selected_event]
if selected_bet_type != "All":
    df_filtered = df_filtered[df_filtered["Bet Type"].astype(str).str.strip() == selected_bet_type]

if "Value" in df_filtered.columns:
    df_filtered = df_filtered.sort_values(by="Value", ascending=False, na_position="last")

styled_filtered = df_filtered.style
if "Value" in df_filtered.columns:
    styled_filtered = styled_filtered.applymap(value_step_style, subset=["Value"])
styled_filtered = styled_filtered.apply(highlight_best_book_cells, axis=1)

# -------------------------
# Helper: Snapshot teaser for free users
# -------------------------
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
# Access (gated UX)
# -------------------------
def _status_banner():
    status, is_sub, email = _entitlement_soft()
    if status == "signed_in" and is_sub:
        st.markdown(f'<div class="tw-banner tw-ok">Signed in as <u>{email}</u>. Subscription is active.</div>', unsafe_allow_html=True)
    elif status == "signed_in" and not is_sub:
        st.markdown(f'<div class="tw-banner tw-warn">Account found for <u>{email}</u>, but no active subscription.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="tw-banner tw-err">You are not signed in.</div>', unsafe_allow_html=True)
    return status, is_sub

def _auth_card():
    st.markdown('<div class="tw-card">', unsafe_allow_html=True)
    st.markdown("### Access")
    status, is_sub = _status_banner()

    # If subscribed → show nothing else in the card (clean)
    if status == "signed_in" and is_sub:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Otherwise show Sign-in + Subscribe sections side-by-side
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Existing subscribers")
        # Narrow container for email field
        st.markdown('<div class="narrow">', unsafe_allow_html=True)
        with st.form(key="signin_form", border=False, clear_on_submit=False):
            email = st.text_input("Email", placeholder="you@example.com", key="signin_email")
            if st.form_submit_button("Sign in", use_container_width=False, type="secondary"):
                if not email:
                    st.error("Enter your email.")
                else:
                    if _dev_login(email.strip()):
                        st.success("Signed in.")
                        st.rerun()
                    else:
                        st.error("No account found or sign-in failed.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown("#### New")
        # Only show subscribe options if NOT already a subscriber
        if not (status == "signed_in" and is_sub):
            pcols = st.columns(2)
            with pcols[0]:
                st.markdown('<div class="tw-plan">', unsafe_allow_html=True)
                st.markdown("**Single Board – $5 / month**")
                with st.form(key="plan_single_form"):
                    email_default = st.session_state.get("email", "")
                    st.markdown('<div class="narrow">', unsafe_allow_html=True)
                    e1 = st.text_input("Email for checkout", value=email_default, placeholder="you@example.com", key="email_p1")
                    st.markdown('</div>', unsafe_allow_html=True)
                    if st.form_submit_button("Subscribe — $5", use_container_width=False):
                        if not e1:
                            st.error("Enter an email to continue.")
                        else:
                            try:
                                url = _checkout_url(e1.strip())
                                st.link_button("Open Stripe Checkout", url)
                            except Exception as ex:
                                st.error(f"Checkout failed: {ex}")
                st.markdown('</div>', unsafe_allow_html=True)

            with pcols[1]:
                st.markdown('<div class="tw-plan">', unsafe_allow_html=True)
                st.markdown("**All Boards – $9 / month**")
                with st.form(key="plan_all_form"):
                    email_default = st.session_state.get("email", "")
                    st.markdown('<div class="narrow">', unsafe_allow_html=True)
                    e2 = st.text_input("Email for checkout", value=email_default, placeholder="you@example.com", key="email_p2")
                    st.markdown('</div>', unsafe_allow_html=True)
                    if st.form_submit_button("Subscribe — $9", use_container_width=False):
                        if not e2:
                            st.error("Enter an email to continue.")
                        else:
                            try:
                                url = _checkout_url(e2.strip())
                                st.link_button("Open Stripe Checkout", url)
                            except Exception as ex:
                                st.error(f"Checkout failed: {ex}")
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Render
# -------------------------
def render_board():
    status, is_sub, _ = _entitlement_soft()

    # If subscriber → full board immediately (no subscribe UI)
    if status == "signed_in" and is_sub:
        _auth_card()  # shows only the green banner (no subscribe UI)
        st.subheader("Full Board")
        st.dataframe(styled_filtered, use_container_width=True, hide_index=True, height=900)
        return

    # Otherwise: show snapshot preview + access card
    st.markdown("### Current Snapshot")
    st.caption("Free preview — top edges with banded EV. Updated every ~10 minutes.")
    st.table(_to_teaser(df_filtered))
    st.info("Subscribe to unlock exact EV%, best book & price, historical movement, and CSV export.")
    _auth_card()
    st.stop()

render_board()
