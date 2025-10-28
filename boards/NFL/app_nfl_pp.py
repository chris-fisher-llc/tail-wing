# app_nfl_pp.py — NFL Player Props Board (polished UI + fixed auth/subscribe UX)

import os
import json
from datetime import datetime, timezone
from typing import Optional, Tuple

import requests
import pandas as pd
import streamlit as st

# ---------------------------
# Config
# ---------------------------
API_BASE = os.getenv("PAYWALL_API", "http://localhost:9000")
BOARD_NAME = os.getenv("BOARD_NAME", "nfl_player_props")  # used in metadata sent to backend
PRICE_ID_SINGLE = os.getenv("PRICE_ID_SINGLE", "")        # e.g. price_123 for $5 single board
PRICE_ID_ALL = os.getenv("PRICE_ID_ALL", "")              # e.g. price_abc for $9 all boards

st.set_page_config(
    page_title="NFL Player Props — Anomaly Board",
    page_icon="🏈",
    layout="wide",
)

# ---------------------------
# Small CSS for hierarchy & visual rhythm
# ---------------------------
st.markdown(
    """
    <style>
      /* Fonts (pairing): Title = Poppins, Section = Montserrat, Body = Inter */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Montserrat:wght@500;600&family=Poppins:wght@600;700&display=swap');

      /* Title */
      .tw-title h1 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
        margin-bottom: 0.35rem;
      }
      .tw-subtitle {
        font-family: 'Montserrat', sans-serif !important;
        font-size: 0.95rem;
        color: #7a7f87;
        margin-bottom: 1.25rem;
      }

      /* Section headers */
      .tw-section h3, .tw-section h4 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem;
      }

      /* Body text */
      .tw-body, .stMarkdown, .stTable, .stDataFrame {
        font-family: 'Inter', sans-serif !important;
      }

      /* Cards */
      .tw-card {
        border: 1px solid #edf0f2;
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        background: #fff;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
      }

      /* Plan cards */
      .tw-plan {
        border: 1px solid #e7ebef;
        border-radius: 16px;
        padding: 16px;
        background: #fcfdff;
      }
      .tw-plan h4{
        margin: 0 0 6px 0;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600;
      }
      .tw-price {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 8px;
      }
      .tw-plan small { color: #70757d; }

      /* Buttons: natural width, strong shape */
      .stButton>button {
        border-radius: 12px;
        padding: 8px 14px;
        font-weight: 600;
      }
      /* Primary */
      .tw-primary .stButton>button {
        background: #1f6feb;
        color: #fff;
        border: 1px solid #1a61ce;
      }
      .tw-primary .stButton>button:hover { filter: brightness(0.96); }

      /* Secondary */
      .tw-secondary .stButton>button {
        background: #f4f7fb;
        color: #0f172a;
        border: 1px solid #e6ebf2;
      }

      /* Status banner */
      .tw-banner {
        border-radius: 12px;
        padding: 10px 12px;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600;
        margin-bottom: 10px;
      }
      .tw-banner.ok { background: #ecfdf3; color: #027a48; border: 1px solid #abefc6; }
      .tw-banner.warn { background: #fff7ed; color: #b45309; border: 1px solid #fed7aa; }
      .tw-banner.err { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }

      /* Subtle divider spacing fixes */
      .block-container { padding-top: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Session helpers
# ---------------------------
def _get_token() -> Optional[str]:
    return st.session_state.get("token")

def _set_token(tok: str):
    st.session_state["token"] = tok

def _get_email() -> Optional[str]:
    return st.session_state.get("email")

def _set_email(email: str):
    st.session_state["email"] = email

def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    tok = _get_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers

# ---------------------------
# Backend calls
# ---------------------------
def dev_sign_in(email: str) -> Tuple[bool, str]:
    """
    Dev-only sign-in that many of your flows use. Falls back to /auth/login if /auth/dev_login isn't present.
    Returns (ok, message).
    """
    endpoints = ["/auth/dev_login", "/auth/login"]
    for ep in endpoints:
        try:
            r = requests.post(
                f"{API_BASE}{ep}",
                headers={"Content-Type": "application/json"},
                json={"email": email},
                timeout=12,
            )
            if r.status_code == 200:
                data = r.json()
                token = data.get("session_token") or data.get("token")
                if token:
                    _set_token(token)
                    _set_email(email)
                    return True, "Signed in."
                else:
                    return False, "Signed in, but no token returned."
            else:
                # keep trying the next endpoint
                continue
        except Exception as e:
            continue
    return False, "Sign-in failed (no auth endpoint responded 200)."

def fetch_entitlement() -> Tuple[bool, str]:
    """
    Query entitlement; if your backend exposes /billing/entitlement, use it.
    Otherwise infer 'has sub' from token presence.
    """
    try:
        r = requests.get(f"{API_BASE}/billing/entitlement", headers=_headers(), timeout=10)
        if r.status_code == 200:
            data = r.json()
            return bool(data.get("is_active", False)), data.get("status", "")
    except Exception:
        pass
    # Fallback: token present but we don't know subscription
    return (_get_token() is not None), ("unknown" if _get_token() else "signed_out")

def start_checkout(price_id: str) -> Tuple[bool, Optional[str], str]:
    """
    Attempts multiple common endpoints; returns (ok, url, debug_message).
    """
    payload = {
        "price_id": price_id,
        "board": BOARD_NAME,
        "email": _get_email()
    }
    endpoints = [
        "/billing/checkout",
        "/billing/create_checkout_session",
        "/billing/checkout_link",
        "/billing/subscribe",
    ]
    debug = []
    for ep in endpoints:
        try:
            r = requests.post(f"{API_BASE}{ep}", headers=_headers(), json=payload, timeout=15)
            if r.status_code == 200:
                data = r.json()
                url = data.get("url") or data.get("checkout_url") or data.get("redirect_url")
                if url:
                    return True, url, f"OK via {ep}"
                else:
                    debug.append(f"{ep}: 200 but no url field")
            else:
                # capture server response text for surfacing 422 payload issues
                try:
                    msg = r.text[:500]
                except Exception:
                    msg = f"HTTP {r.status_code}"
                debug.append(f"{ep}: {msg}")
        except Exception as e:
            debug.append(f"{ep}: {repr(e)}")
    return False, None, " | ".join(debug[:3])

# ---------------------------
# UI Sections
# ---------------------------
def header_section():
    st.markdown('<div class="tw-title">', unsafe_allow_html=True)
    st.title("NFL Player Props — Anomaly Board 🏈")
    st.markdown(
        '<div class="tw-subtitle">Powered by The Tail Wing — scanning books for alt-yardage & anytime TD edges</div></div>',
        unsafe_allow_html=True,
    )
    top_bar = st.container()
    with top_bar:
        cols = st.columns([1, 3, 1])
        with cols[1]:
            with st.container():
                st.markdown('<div class="tw-secondary">', unsafe_allow_html=True)
                st.button("Refresh Odds", key="refresh_odds_btn")  # hook into your refresh handler if needed
                st.markdown("</div>", unsafe_allow_html=True)

def snapshot_section(df: Optional[pd.DataFrame] = None):
    st.markdown('<div class="tw-section">', unsafe_allow_html=True)
    st.subheader("Current Snapshot")
    st.caption("Free preview — top edges with baked EV. Updated every ~10 minutes.")
    if df is None or df.empty:
        st.info("Snapshot loads here. (Hook your existing dataframe into `snapshot_section(df)`.)")
    else:
        st.dataframe(df, use_container_width=True, height=350)
    st.markdown("</div>", unsafe_allow_html=True)

def status_banner():
    is_active, status = fetch_entitlement()
    email = _get_email()
    if email and is_active:
        st.markdown(
            f'<div class="tw-banner ok">Signed in as <u>{email}</u>. Subscription is active.</div>',
            unsafe_allow_html=True,
        )
    elif email and not is_active:
        st.markdown(
            f'<div class="tw-banner warn">Account found for <u>{email}</u>, but no active subscription.</div>',
            unsafe_allow_html=True,
        )
    elif not email:
        st.markdown(
            '<div class="tw-banner err">You are not signed in.</div>',
            unsafe_allow_html=True,
        )

def auth_and_subscribe_section():
    st.markdown('<div class="tw-card">', unsafe_allow_html=True)
    st.markdown("### Access")
    status_banner()

    left, right = st.columns([1, 1])

    # Left: Sign in
    with left:
        st.markdown("#### Existing subscribers")
        with st.form(key="signin_form"):
            email = st.text_input("Email", key="signin_email", placeholder="you@example.com")
            submitted = st.form_submit_button("Sign in")
            if submitted:
                ok, msg = dev_sign_in(email.strip())
                if ok:
                    st.success("Signed in.")
                else:
                    st.error(msg)

    # Right: Subscribe
    with right:
        st.markdown("#### New")
        # Two plan cards side-by-side
        pcols = st.columns(2)
        with pcols[0]:
            st.markdown('<div class="tw-plan">', unsafe_allow_html=True)
            st.markdown("<h4>Single Board</h4>", unsafe_allow_html=True)
            st.markdown('<div class="tw-price">$5 / month</div>', unsafe_allow_html=True)
            st.markdown("<small>Unlock this NFL board.</small>", unsafe_allow_html=True)
            with st.form(key="plan_single_form"):
                submit = st.form_submit_button("Subscribe — $5")
                if submit:
                    if not PRICE_ID_SINGLE:
                        st.error("PRICE_ID_SINGLE not set.")
                    else:
                        ok, url, debug = start_checkout(PRICE_ID_SINGLE)
                        if ok and url:
                            st.link_button("Open Checkout", url, help="Opens Stripe checkout in a new tab.")
                        else:
                            st.error("Failed to start checkout.")
                            st.caption(debug)
            st.markdown("</div>", unsafe_allow_html=True)

        with pcols[1]:
            st.markdown('<div class="tw-plan">', unsafe_allow_html=True)
            st.markdown("<h4>All Boards</h4>", unsafe_allow_html=True)
            st.markdown('<div class="tw-price">$9 / month</div>', unsafe_allow_html=True)
            st.markdown("<small>Access NFL, NBA, CFB, NHL boards (as available).</small>", unsafe_allow_html=True)
            with st.form(key="plan_all_form"):
                submit = st.form_submit_button("Subscribe — $9")
                if submit:
                    if not PRICE_ID_ALL:
                        st.error("PRICE_ID_ALL not set.")
                    else:
                        ok, url, debug = start_checkout(PRICE_ID_ALL)
                        if ok and url:
                            st.link_button("Open Checkout", url, help="Opens Stripe checkout in a new tab.")
                        else:
                            st.error("Failed to start checkout.")
                            st.caption(debug)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end card

# ---------------------------
# Main
# ---------------------------
def run_app():
    header_section()

    # TODO: wire in your real snapshot dataframe here:
    # render_df = <your function>()
    render_df = None  # placeholder to keep this file drop-in friendly
    snapshot_section(render_df)

    auth_and_subscribe_section()

if __name__ == "__main__":
    run_app()
