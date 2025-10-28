import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import pytz
import requests
from urllib.parse import urlencode  # needed for _subscribe()

# ---- PAGE SETUP (do this first) ----
st.set_page_config(page_title="The Tail Wing - NFL Player Props", layout="wide")

# ---- Config ----
PAYWALL_API  = st.secrets["PAYWALL_API"]
FRONTEND_URL = st.secrets["FRONTEND_URL"]
PRICE_ALL    = st.secrets["STRIPE_PRICE_ALL"]
PRICE_SINGLE = st.secrets.get("STRIPE_PRICE_SINGLE", PRICE_ALL)  # fallback for now

# Keep a single base for all auth/billing calls
API_BASE = PAYWALL_API

# ---- Auth helpers ----
def _auth_headers():
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}

def _redirect(url: str):
    # Simple client-side redirect
    st.markdown(f'<meta http-equiv="refresh" content="0; url={url}">', unsafe_allow_html=True)
    st.stop()

def _subscribe(price_id: str, board_key: str = "all"):
    """Create a Stripe Checkout session via backend and redirect the user."""
    try:
        payload = {
            "email": st.session_state.get("email"),
            "referrer": None,
        }
        r = requests.post(f"{PAYWALL_API}/billing/checkout",
                          json=payload, headers=_auth_headers(), timeout=20)
        r.raise_for_status()
        _redirect(r.json()["checkout_url"])
    except Exception as e:
        st.error(f"Checkout failed: {e}")

def _open_portal():
    """Open Stripe Billing Portal for the current user."""
    try:
        payload = {"return_path": "?"}  # back to board root after portal
        r = requests.post(f"{PAYWALL_API}/billing/create_portal_session",
                          json=payload, headers=_auth_headers(), timeout=20)
        r.raise_for_status()
        _redirect(r.json()["portal_url"])
    except Exception as e:
        st.error(f"Billing portal failed: {e}")

# ---- Dev sign-in (keep for now) ----
def _dev_sign_in():
    st.subheader("Sign in to unlock full board")
    email = st.text_input("Email", placeholder="you@example.com")
    if st.button("Sign in (dev)"):
        r = requests.post(f"{API_BASE}/auth/dev_login", json={"email": email}, timeout=10)
        r.raise_for_status()
        st.session_state["token"] = r.json()["session_token"]
        st.session_state["email"] = email 
        st.rerun()
    st.stop()

# ---- Entitlement check (returns is_subscriber flag + headers) ----
def _entitlement():
    if "token" not in st.session_state:
        _dev_sign_in()
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}
    r = requests.get(f"{API_BASE}/entitlement", headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    st.caption(f"Signed in as: {data.get('email','')} • Subscriber: {'Yes' if data.get('is_subscriber') else 'No'}")
    return bool(data.get("is_subscriber")), headers

# ---- Teaser utility ----
def _to_teaser(df: pd.DataFrame) -> pd.DataFrame:
    safe_cols = [c for c in df.columns if c.lower() not in {"ev%", "ev", "best book", "best_book", "best price", "best_price", "price"}]
    out = df[safe_cols].copy()
    ev_col = next((c for c in df.columns if c.lower() in {"ev%", "ev", "value"}), None)
    if ev_col:
        def band(x):
            try:
                v = float(x)
            except Exception:
                return "—"
            if v < 2: return "<2%"
            if v < 4: return "2–4%"
            if v < 6: return "4–6%"
            if v < 9: return "6–9%"
            return "9%+"
        out["Edge Band"] = pd.to_numeric(df[ev_col], errors="coerce").map(band)
    return out.head(3)

# ---- Main board renderer (handles paywall gating) ----
def render_board(df_full: pd.DataFrame, styled_full):
    """Render preview (free) vs full (subscriber) while preserving styling."""
    is_subscriber, headers = _entitlement()

    st.markdown("### Current Snapshot")

    if not is_subscriber:
        # Preview content
        st.write("**Free preview** — top edges with banded EV. Updated every ~10 minutes.")
        st.table(_to_teaser(df_full))
        st.info("Subscribe to unlock exact EV%, best book & price, historical movement, and CSV export.")

        # Subscription actions
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Subscribe – All Boards (monthly)"):
                _subscribe(PRICE_ALL, board_key="all")
        # with c2:
        #     st.button("Manage Billing", on_click=_open_portal)

        # Dev tools
        with st.expander("Developer tools (local only)"):
            if st.button("Simulate subscription (dev)"):
                requests.post(
                    f"{API_BASE}/dev/toggle_subscription",
                    headers=headers,
                    json={"is_subscriber": True},
                    timeout=10
                )
                st.rerun()

        st.stop()

    # Subscriber view
    st.success("Subscriber view: full board unlocked.")
    st.dataframe(styled_full, use_container_width=True, hide_index=True, height=1200)
    st.caption("Watermark: For your account • Live at HH:MM")

    with st.expander("Developer tools (local only)"):
        if st.button("Turn off subscription (dev)"):
            requests.post(
                f"{API_BASE}/dev/toggle_subscription",
                headers=headers,
                json={"is_subscriber": False},
                timeout=10
            )
            st.rerun()

# ---- Handle return from Stripe Checkout (new API) ----
_qp = st.query_params            # property, not a function
if _qp.get("checkout") == "success":
    st.success("Subscription activated. Loading full board…")
    st.query_params.clear()      # wipe query params

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

# ---- Manual refresh (GitHub Actions trigger) ----
def trigger_github_action():
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")  # e.g., "chris-fisher-llc/tail-wing"
    workflow_file = st.secrets.get("GITHUB_WORKFLOW_FILE", "update-nfl-player-props.yml")
    ref = st.secrets.get("GITHUB_REF", "main")

    if not token or not repo:
        st.error("Missing secrets: please set GITHUB_TOKEN and GITHUB_REPO in st.secrets.")
        return

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

# ---- CSV path resolution ----
def _find_csv_path() -> Path | None:
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

# ---- Refresh button ----
btn_cols = st.columns([1, 1, 1])
with btn_cols[1]:
    if st.button("Refresh Odds", use_container_width=True):
        trigger_github_action()
        st.info("Waiting for new data...")
        wait_for_csv_update()

# ---- Main App ----
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

    if "Bet Type" in df.columns:
        df["Bet Type"] = df["Bet Type"].astype(str).str.strip()

    fixed_cols = {"Event", "Player", "Bet Type", "Alt Line", "Best Book", "Best Odds", "Value",
                  "best_decimal", "avg_other"}
    odds_cols = [c for c in df.columns if c not in fixed_cols and not str(c).startswith("Unnamed")]

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

    df["Value"] = pd.to_numeric(df.get("Value"), errors="coerce")
    df["_Value_print"] = df["Value"].map(lambda x: f"{x:.3f}".rstrip("0").rstrip(".") if pd.notnull(x) else "")

    display_cols = ["Event", "Player", "Bet Type", "Alt Line"] + odds_cols + ["Value", "_Value_print", "Best Book", "Best Odds"]
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols].copy()

    # ---- Sidebar filters ----
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

    # -------- Styling --------
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

    # ---- Paywall render (preserves your styling) ----
    render_board(render_df, styled)

# Run if executed directly
if __name__ == "__main__":
    run_app()
