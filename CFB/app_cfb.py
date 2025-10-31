import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import pytz
import requests
import time
import math
import re

st.set_page_config(page_title="The Tail Wing - CFB Games", layout="wide")

# --- Clear caches (helps stuck sessions) ---
try:
    st.cache_data.clear(); st.cache_resource.clear()
except Exception:
    pass

# ---- Header ----
st.markdown(
    """
    <h1 style='text-align: center; font-size: 42px;'>
       🏈 College Football — Game Lines
    </h1>
    <p style='text-align: center; font-size:18px; color: gray;'>
        Game Spreads · Moneylines · Totals (pair-first fair pricing with vig-curve fallback)
    </p>
    """,
    unsafe_allow_html=True
)

# ---- Mobile hint (?mobile=1) ----
components.html(
    """
    <script>
      const w = Math.min(window.innerWidth || 9999, screen.width || 9999);
      const isMobile = w < 800;
      try {
        const url = new URL(window.location);
        if (isMobile) url.searchParams.set('mobile','1');
        else url.searchParams.delete('mobile');
        window.history.replaceState({},'',url);
      } catch(e){}
    </script>
    """,
    height=0,
)

# One-shot refresh to pick up ?mobile
try:
    st.session_state.setdefault("_awaited_mobile_param_cfb", False)
    if not st.session_state["_awaited_mobile_param_cfb"]:
        st.session_state["_awaited_mobile_param_cfb"] = True
        st.autorefresh(interval=250, limit=1, key="await_mobile_param_cfb")
except Exception:
    pass

# ---- GitHub Action trigger (optional) ----
def trigger_github_action():
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")          # e.g. "chris-fisher-llc/tail-wing"
    workflow_file = st.secrets.get("GITHUB_WORKFLOW_FILE", "main_cfb.yml")
    ref = st.secrets.get("GITHUB_REF", "main")
    if not token or not repo:
        st.error("Missing secrets: GITHUB_TOKEN / GITHUB_REPO.")
        return False
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"ref": ref}
    with st.spinner("Triggering GitHub Action…"):
        r = requests.post(url, headers=headers, json=payload, timeout=15)
    if r.status_code == 204:
        st.success("Refresh kicked off. Board will auto-update when CSV is pushed.")
        return True
    st.error(f"Workflow start failed ({r.status_code}): {r.text}")
    return False

def wait_for_csv_update(max_checks=12, sleep_seconds=10):
    csv_path = _find_csv_path()
    if not csv_path or not csv_path.exists(): return
    old_mtime = csv_path.stat().st_mtime
    with st.spinner("Waiting for new data…"):
        for _ in range(max_checks):
            time.sleep(sleep_seconds)
            if csv_path.exists() and csv_path.stat().st_mtime != old_mtime:
                st.success("Data updated — reloading!")
                st.rerun()

def _find_csv_path() -> Path | None:
    env = os.getenv("CFB_GAMES_CSV")
    if env:
        p = Path(env)
        if p.exists(): return p
    here = Path(__file__).resolve().parent
    candidates = [
        here / "cfb_games.csv",
        here / "cfb" / "cfb_games.csv",
        here.parent / "cfb_games.csv",
        here.parent / "cfb" / "cfb_games.csv",
        Path.cwd() / "cfb_games.csv",
        Path.cwd() / "cfb" / "cfb_games.csv",
    ]
    for p in candidates:
        if p.exists(): return p
    try:
        for p in here.rglob("cfb_games.csv"):
            return p
    except Exception:
        pass
    return None

# ---- Top button row ----
c1,c2,c3 = st.columns([1,1,1])
with c2:
    if st.button("Refresh Odds", use_container_width=True):
        if trigger_github_action():
            wait_for_csv_update()

# ---------- Helpers ----------
def to_american(x):
    try:
        x = int(float(x))
        return f"+{x}" if x > 0 else str(x)
    except Exception:
        return ""

def american_to_prob(o: float) -> float:
    try:
        o = float(o)
        return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)
    except Exception:
        return float("nan")

def prob_to_decimal(p: float) -> float:
    return 1/p if p and p>0 else float("nan")

# pooled vig curve coeffs (+ 10% floor)
A,B,C = -33.854, 0.3313, 0.0001
def apply_vig_curve(o: float) -> float:
    m = abs(o)
    vig = max(A + B*m + C*(m**2), 0.10*m)
    return m + vig if o >= 0 else -(m - vig)

OU_RE = re.compile(r'\b(Over|Under)\b', re.I)

def normalize_market(bt: str) -> str:
    s = str(bt).strip().lower()
    if "total" in s or "over" in s or "under" in s: return "Total"
    if "money" in s: return "Moneyline"
    return "Spread"

def infer_side(row) -> str | None:
    # totals: Over/Under from bet_type text
    bt = str(row.get("Bet Type",""))
    m = OU_RE.search(bt)
    if m:
        return m.group(1).title()
    # moneyline/spread: use selection if present
    for key in ("selection","Selection","team","Team"):
        if key in row.index and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
    return None

def pair_midpoint_for_book(df: pd.DataFrame, book_col: str, event: str, market: str,
                           line_val, side_label: str, offered: float) -> float | None:
    """
    Look for the *opposite side* at the same book for the same game/market/line.
    - Moneyline: same game/market; side is opponent team
    - Spread: same game/market and |line| equal; side is opponent team
    - Total: same game/market and identical line; side is Over<->Under
    Return fair American odds via midpoint if found; else None.
    """
    # derive opponent label
    def opponent_of(side: str) -> str | None:
        if market == "Total":
            return "Under" if side and side.lower()=="over" else ("Over" if side and side.lower()=="under" else None)
        # team vs opponent from game string "Away @ Home"
        if " @ " in str(event):
            away, home = [x.strip() for x in str(event).split(" @ ", 1)]
            if side and side.lower()==away.lower(): return home
            if side and side.lower()==home.lower(): return away
        # fallback: just pick a different side within the game
        return None

    opp_side = opponent_of(side_label)

    pool = df[(df["Event"]==event) & (df["Market"]==market)]
    if market in ("Spread","Total"):
        # pair by exact total line, or absolute spread line
        try:
            if market == "Total":
                pool = pool[pd.to_numeric(pool["Alt Line"], errors="coerce") == pd.to_numeric(line_val, errors="coerce")]
            else:
                pool = pool[pool["|Alt|"] == abs(float(line_val))]
        except Exception:
            pass

    opp_odds = None
    for _, r in pool.iterrows():
        if r.name == event:  # harmless guard
            pass
        # if we know the explicit opposite side, require it
        r_side = str(r.get("_side","")).strip()
        if opp_side:
            if r_side.lower() != opp_side.lower(): 
                continue
        else:
            # else: any *different* side within pool
            if r_side.lower() == str(side_label or "").lower():
                continue
        v = r.get(book_col, None)
        if pd.notna(v) and str(v).strip()!="":
            try:
                opp_odds = float(v); break
            except Exception:
                pass

    if opp_odds is None:
        return None

    m = (abs(offered) + abs(opp_odds)) / 2.0
    return m if offered >= 0 else -m

# ---------- App core ----------
def run_app(df: pd.DataFrame | None = None):
    qp = st.query_params
    raw_mobile = qp.get("mobile", None)
    auto_mobile = "1" in raw_mobile if isinstance(raw_mobile, list) else raw_mobile == "1"

    # Load CSV
    if df is None:
        csv_path = _find_csv_path()
        if not csv_path or not csv_path.exists():
            st.error("cfb_games.csv not found.")
            return
        try:
            df = pd.read_csv(csv_path)
            df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed')]
            df = df.dropna(axis=1, how="all")
            ts = datetime.fromtimestamp(csv_path.stat().st_mtime, pytz.utc)
            eastern = ts.astimezone(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %I:%M %p %Z")
            st.caption(f"Odds last updated: {eastern}")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return

    if df.empty:
        st.warning("No data to display."); return

    # --- Normalize field names from your schema ---
    df = df.rename(columns={
        "game":"Event",
        "bet_type":"Bet Type",
        "line":"Alt Line",
        "selection":"Selection",
        "opponent":"Opponent",
        "best_book":"Best Book",
        "best_odds":"Best Odds",
        "value_ratio":"Line vs. Average",
    })

    # Rename sportsbook columns: drop trailing "_Odds"
    new_cols = {}
    for c in df.columns:
        if c.endswith("_Odds"):
            new_cols[c] = c[:-5]  # strip "_Odds"
    df = df.rename(columns=new_cols)

    # Compute Market normalization
    df["Market"] = df["Bet Type"].apply(normalize_market)

    # Dynamic sportsbook columns
    fixed = {"Event","Bet Type","Market","Alt Line","Selection","Opponent",
             "Best Book","Best Odds","Line vs. Average","kickoff_et"}
    book_cols = [c for c in df.columns if c not in fixed and not str(c).startswith("Unnamed")]

    # Count #books
    def _is_valid_num(v): return pd.notnull(v) and str(v).strip() != ""
    if book_cols:
        df["# Books"] = df[book_cols].apply(lambda r: sum(_is_valid_num(x) for x in r.values), axis=1)
    else:
        df["# Books"] = 0

    # Sidebar
    with st.sidebar:
        st.header("Filters")
        evs = ["All"] + sorted(df["Event"].dropna().unique())
        sel_event = st.selectbox("Game", evs, 0)
        markets = ["All"] + sorted(df["Market"].dropna().unique())
        sel_market = st.selectbox("Market", markets, 0)
        books = ["All"] + sorted(df["Best Book"].dropna().unique()) if "Best Book" in df.columns else ["All"]
        sel_book = st.selectbox("Best Book", books, 0)
        min_books = st.number_input("Min. books posting this line", 1, 20, 2, 1)
        compact_mobile = st.toggle("Compact mobile mode", value=bool(auto_mobile))

    if sel_event!="All": df=df[df["Event"]==sel_event]
    if sel_market!="All": df=df[df["Market"]==sel_market]
    if sel_book!="All": df=df[df["Best Book"]==sel_book]
    df=df[df["# Books"]>=int(min_books)]

    # Format sportsbook odds as American strings for display later
    def fmt_american_cell(x):
        try:
            x = int(float(x))
            return f"+{x}" if x > 0 else str(x)
        except Exception:
            return ""
    for col in book_cols: df[col] = df[col].apply(fmt_american_cell)
    if "Best Odds" in df.columns: df["Best Odds"] = df["Best Odds"].apply(fmt_american_cell)

    # Percentify "Line vs. Average"
    df["Line vs. Average"] = pd.to_numeric(df.get("Line vs. Average"), errors="coerce")
    df["Line vs. Average (%)"] = (df["Line vs. Average"] - 1.0) * 100.0

    # Side inference and keys for pairing
    df["_side"] = df.apply(infer_side, axis=1)
    # absolute spread line key for pairing
    def _abs_key(row):
        try:
            return abs(float(row.get("Alt Line")))
        except Exception:
            return None
    df["|Alt|"] = df.apply(_abs_key, axis=1)

    # ---------- Implied EV: pair-first, curve-fallback ----------
    def fair_for_other_book(book: str, offered: float, row: pd.Series) -> float:
        # first try to find opposite side in same book
        fair = pair_midpoint_for_book(
            df=df,
            book_col=book,
            event=row.get("Event"),
            market=row.get("Market"),
            line_val=row.get("Alt Line"),
            side_label=row.get("_side"),
            offered=offered
        )
        if fair is not None:
            return fair
        # fallback: apply vig curve
        return apply_vig_curve(offered)

    def calc_implied_ev(row: pd.Series) -> float:
        try:
            best_odds = float(row.get("Best Odds", float("nan")))
            if not math.isfinite(best_odds): return float("nan")
            best_book = row.get("Best Book")

            # gather other book odds for same row
            others = []
            for col in book_cols:
                if col == best_book:  # don't adjust the best line
                    continue
                v = row.get(col, None)
                try:
                    if pd.notna(v) and str(v).strip()!="":
                        others.append((col, float(v)))
                except Exception:
                    pass
            if not others: return float("nan")

            # compute fair probs for other books
            fair_probs = []
            for book, offered in others:
                fair_american = fair_for_other_book(book, offered, row)
                p = american_to_prob(fair_american)
                if math.isfinite(p) and p > 0:
                    fair_probs.append(p)
            if not fair_probs: return float("nan")

            avg_prob = sum(fair_probs) / len(fair_probs)
            true_decimal = prob_to_decimal(avg_prob)

            best_decimal = prob_to_decimal(american_to_prob(best_odds))
            if not (math.isfinite(best_decimal) and math.isfinite(true_decimal)):
                return float("nan")

            return (best_decimal / true_decimal - 1.0) * 100.0
        except Exception:
            return float("nan")

    df["Implied EV (%)"] = df.apply(calc_implied_ev, axis=1)

    # ---------- Build render dataframe ----------
    base_cols = ["Event","Bet Type","Alt Line","Selection","Opponent"]
    is_mobile = bool(auto_mobile or compact_mobile)

    # keep Best Book visible so we can shade the winning sportsbook cell
    if sel_book!="All" and is_mobile:
        show_cols = [sel_book] if sel_book in book_cols else []
    else:
        show_cols = book_cols.copy()

    hidden = {"# Books", "Best Odds"}  # hide Best Odds from display
    cols = base_cols + show_cols + ["Best Book","Line vs. Average (%)","Implied EV (%)"]
    cols = [c for c in cols if c in df.columns and c not in hidden]
    render_df = df[cols].copy()

    # compact cosmetics
    def _shorten(x,maxlen=22):
        s=str(x); return (s[:maxlen-1]+"…") if len(s)>maxlen else s
    if is_mobile:
        if "Event" in render_df:  render_df["Event"]=_shorten
        render_df["Event"] = render_df["Event"].apply(lambda s:_shorten(s,22))
        if "Selection" in render_df: render_df["Selection"]=render_df["Selection"].apply(lambda s:_shorten(s,18))
        if "Opponent" in render_df:  render_df["Opponent"]=render_df["Opponent"].apply(lambda s:_shorten(s,18))

    render_df = render_df.sort_values(by=["Implied EV (%)"], ascending=False, na_position="last")

    # ---------- Styling ----------
    def _ev_green(val):
        try: v=float(val)
        except: return ""
        if not math.isfinite(v) or v<=0: return ""
        cap=20.0; alpha=0.15+0.8*min(v/cap,1.0)
        return f"background-color: rgba(34,139,34,{alpha}); font-weight:600;"

    def _shade_best_book(row):
        styles=[""]*len(row)
        bb=row.get("Best Book","")
        shade=_ev_green(row.get("Implied EV (%)",0))
        if bb and bb in row.index:
            try:
                idx=list(row.index).index(bb)
                styles[idx]=shade
            except: pass
        return styles

    if is_mobile:
        st.markdown("""
        <style>
          [data-testid="stDataFrame"] * {font-size:0.92rem!important;}
          .stDataFrame tbody tr td:nth-child(1),
          .stDataFrame thead tr th:nth-child(1){
            max-width:200px!important;white-space:nowrap!important;
            text-overflow:ellipsis!important;overflow:hidden!important;}
        </style>""", unsafe_allow_html=True)

    styled = render_df.style
    styled = styled.applymap(_ev_green, subset=["Implied EV (%)"])
    styled = styled.apply(_shade_best_book, axis=1)
    styled = styled.format({
        "Line vs. Average (%)": "{:.1f}%",
        "Implied EV (%)": "{:.1f}%"
    })
    styled = styled.set_table_styles([{
        'selector':'th','props':[('font-weight','bold'),('text-align','center'),
        ('font-size','16px'),('background-color','#003366'),('color','white')]
    }])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

# ---- Run ----
if __name__=="__main__":
    run_app()
