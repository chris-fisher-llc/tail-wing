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

# --- Best-effort mobile hint ---
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

# --- One-shot refresh for ?mobile param ---
try:
    st.session_state.setdefault("_awaited_mobile_param", False)
    if not st.session_state["_awaited_mobile_param"]:
        st.session_state["_awaited_mobile_param"] = True
        st.autorefresh(interval=250, limit=1, key="await_mobile_param")
except Exception:
    pass

# ---------- GitHub Action Trigger ----------
def trigger_github_action():
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")
    workflow_file = st.secrets.get("GITHUB_WORKFLOW_FILE", "main_nba.yml")
    ref = st.secrets.get("GITHUB_REF", "main")
    if not token or not repo:
        st.error("Missing secrets: please set GITHUB_TOKEN and GITHUB_REPO.")
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
        st.success("Refresh kicked off. Odds will update automatically.")
        return True
    st.error(f"Failed to trigger workflow ({r.status_code}): {r.text}")
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

# ---------- CSV discovery ----------
def _find_csv_path() -> Path | None:
    env = os.getenv("NBA_PROPS_CSV")
    if env:
        p = Path(env)
        if p.exists(): return p
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
        if p.exists(): return p
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
btn_cols = st.columns([1,1,1])
with btn_cols[1]:
    if st.button("Refresh Odds", use_container_width=True):
        if trigger_github_action():
            wait_for_csv_update()

# ---------- Core ----------
def run_app(df: pd.DataFrame | None = None):
    qp = st.query_params
    raw_mobile = qp.get("mobile", None)
    auto_mobile = "1" in raw_mobile if isinstance(raw_mobile, list) else raw_mobile == "1"

    if df is None:
        csv_path = _find_csv_path()
        if not csv_path or not csv_path.exists():
            st.error("nba_player_props.csv not found.")
            return
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed')]
        df = df.dropna(axis=1, how="all")
        ts = datetime.fromtimestamp(csv_path.stat().st_mtime, pytz.utc)
        eastern = ts.astimezone(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %I:%M %p %Z")
        st.caption(f"Odds last updated: {eastern}")

    if df.empty:
        st.warning("No data to display."); return

    # --- Rename and setup
    df = df.rename(columns={
        "event":"Event","player":"Player","group":"Bet Type","threshold":"Alt Line",
        "best_book":"Best Book","best_odds":"Best Odds","value_ratio":"Line vs. Average"
    })

    fixed_cols_raw = {"Event","Player","Bet Type","Alt Line","Best Book","Best Odds",
                      "Line vs. Average","best_decimal","avg_other"}
    book_cols = [c for c in df.columns if c not in fixed_cols_raw and not str(c).startswith("Unnamed")]

    # Count books posting
    def _is_valid_num(v): return pd.notnull(v) and str(v).strip() != ""
    df["# Books"] = df[book_cols].apply(lambda r: sum(_is_valid_num(x) for x in r.values), axis=1)

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        evs = ["All"] + sorted(df["Event"].dropna().unique()) if "Event" in df.columns else ["All"]
        sel_event = st.selectbox("Event", evs, 0)
        bets = ["All"] + sorted(df["Bet Type"].dropna().unique()) if "Bet Type" in df.columns else ["All"]
        sel_bet = st.selectbox("Bet Type", bets, 0)
        books = ["All"] + sorted(df["Best Book"].dropna().unique()) if "Best Book" in df.columns else ["All"]
        sel_book = st.selectbox("Best Book", books, 0)
        min_books = st.number_input("Min. books posting this line",1,20,4,1)
        compact_mobile = st.toggle("Compact mobile mode", value=bool(auto_mobile))

    if sel_event!="All": df=df[df["Event"]==sel_event]
    if sel_bet!="All": df=df[df["Bet Type"].astype(str).str.strip()==sel_bet]
    if sel_book!="All": df=df[df["Best Book"]==sel_book]
    df=df[df["# Books"]>=int(min_books)]

    for col in book_cols: df[col]=df[col].apply(to_american)
    if "Best Odds" in df.columns: df["Best Odds"]=df["Best Odds"].apply(to_american)
    df["Line vs. Average"]=pd.to_numeric(df["Line vs. Average"],errors="coerce")
    df["Line vs. Average (%)"]=(df["Line vs. Average"]-1.0)*100.0

    # ---------- Implied EV Calculation ----------
    def american_to_prob(o):
        try:
            o=float(o)
            return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)
        except Exception:
            return float("nan")

    def prob_to_decimal(p): return 1/p if p>0 else float("nan")

    A,B,C=-33.854,0.3313,0.0001
    def apply_vig_curve(o):
        m=abs(o)
        vig=max(A+B*m+C*m**2,0.1*m)
        return m+vig if o>=0 else -(m-vig)

    def calc_implied_ev(row):
        try:
            best_odds=float(row.get("Best Odds",float("nan")))
            if not math.isfinite(best_odds): return float("nan")
            best_book=row.get("Best Book")
            others=[]
            for col in book_cols:
                if col==best_book: continue
                v=row.get(col,None)
                try:
                    if pd.notna(v) and str(v).strip()!="": others.append(float(v))
                except: pass
            if not others: return float("nan")
            fair_probs=[]
            for o in others:
                fair_american=apply_vig_curve(o)
                p=american_to_prob(fair_american)
                if math.isfinite(p) and p>0: fair_probs.append(p)
            if not fair_probs: return float("nan")
            avg_prob=sum(fair_probs)/len(fair_probs)
            true_decimal=prob_to_decimal(avg_prob)
            best_decimal=prob_to_decimal(american_to_prob(best_odds))
            return (best_decimal/true_decimal-1)*100
        except Exception:
            return float("nan")

    df["Implied EV (%)"]=df.apply(calc_implied_ev,axis=1)
    base_cols=["Event","Player","Bet Type","Alt Line"]
    is_mobile=bool(auto_mobile or compact_mobile)

    # keep Best Book visible so we can shade the winning sportsbook cell
    if sel_book!="All" and is_mobile:
        show_cols = [sel_book] if sel_book in book_cols else []
    else:
        show_cols = book_cols.copy()
    
    hidden = {"# Books", "Best Odds"}  # <-- do NOT hide "Best Book"
    
    cols = ["Event","Player","Bet Type","Alt Line"] + show_cols + [
        "Best Book",                # <-- include
        "Line vs. Average (%)",
        "Implied EV (%)"
    ]
    cols = [c for c in cols if c in df.columns and c not in hidden]
    
    render_df = df[cols].copy()

    # Compact cosmetics
    def _shorten(x,maxlen=18):
        s=str(x)
        return (s[:maxlen-1]+"…") if len(s)>maxlen else s
    if is_mobile:
        if "Event" in render_df: render_df["Event"]=render_df["Event"].apply(lambda s:_shorten(s,18))
        if "Player" in render_df: render_df["Player"]=render_df["Player"].apply(lambda s:_shorten(s,16))

    render_df=render_df.sort_values(by=["Implied EV (%)"],ascending=False,na_position="last")

    # ---------- Table Styling ----------
    def _ev_green(v):
        try:v=float(v)
        except:return""
        if not math.isfinite(v) or v<=0:return""
        cap=20.0; alpha=0.15+0.8*min(v/cap,1.0)
        return f"background-color: rgba(34,139,34,{alpha}); font-weight:600;"

    def _shade_best_book(row):
        styles=[""]*len(row)
        bb=row.get("Best Book",None)
        if bb and bb in row.index:
            shade=_ev_green(row.get("Implied EV (%)",0))
            try:styles[list(row.index).index(bb)]=shade
            except:pass
        return styles

    if is_mobile:
        st.markdown("""
        <style>
          [data-testid="stDataFrame"] * {font-size:0.92rem!important;}
          .stDataFrame tbody tr td:nth-child(1),
          .stDataFrame thead tr th:nth-child(1),
          .stDataFrame tbody tr td:nth-child(2),
          .stDataFrame thead tr th:nth-child(2){
            max-width:140px!important;white-space:nowrap!important;
            text-overflow:ellipsis!important;overflow:hidden!important;}
        </style>""",unsafe_allow_html=True)

    styled=render_df.style
    styled=styled.applymap(_ev_green,subset=["Implied EV (%)"])
    styled=styled.apply(_shade_best_book,axis=1)
    styled=styled.format({"Line vs. Average (%)":"{:.1f}%","Implied EV (%)":"{:.1f}%"})
    styled=styled.set_table_styles([{
        'selector':'th','props':[('font-weight','bold'),('text-align','center'),
        ('font-size','16px'),('background-color','#003366'),('color','white')]
    }])

    st.dataframe(styled,use_container_width=True,hide_index=True,height=1200)

if __name__=="__main__":
    run_app()
