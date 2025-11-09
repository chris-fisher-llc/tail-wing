# NFL/app_nfl_pp.py — NFL Player Props with per-book vig curves
import os, json, math, re, time
from pathlib import Path
from datetime import datetime

import pandas as pd
import pytz
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="The Tail Wing - NFL Player Props", layout="wide")

# Clear caches to avoid stale sessions on rerun
try:
    st.cache_data.clear(); st.cache_resource.clear()
except Exception:
    pass

# ==========================
# Header
# ==========================
st.markdown("""
<h1 style='text-align:center; font-size:42px;'>NFL — Player Props</h1>
<p style='text-align:center; color:gray; font-size:18px;'>
Pair-first fair pricing with per-book vig curves (fallback)
</p>
""", unsafe_allow_html=True)

# ==========================
# Mobile param bootstrap
# ==========================
components.html("""
<script>
  const w=Math.min(window.innerWidth||9999,screen.width||9999);
  const isMobile=w<800;
  try{
    const url=new URL(window.location);
    if(isMobile) url.searchParams.set('mobile','1');
    else url.searchParams.delete('mobile');
    window.history.replaceState({},'',url);
  }catch(e){}
</script>
""", height=0)

try:
    st.session_state.setdefault("_awaited_mobile_param_nfl", False)
    if not st.session_state["_awaited_mobile_param_nfl"]:
        st.session_state["_awaited_mobile_param_nfl"] = True
        st.autorefresh(interval=250, limit=1, key="await_mobile_param_nfl")
except Exception:
    pass

# ==========================
# Refresh (GitHub Actions) support
# ==========================
def _find_csv_path() -> Path | None:
    env = os.getenv("nfl_matched_output_CSV")
    if env:
        p = Path(env)
        if p.exists(): return p
    here = Path(__file__).resolve().parent
    candidates = [
        here / "nfl_matched_output.csv",
        here.parent / "nfl_matched_output.csv",
        Path.cwd() / "nfl_matched_output.csv",
    ]
    for p in candidates:
        if p.exists(): return p
    try:
        for p in here.rglob("nfl_matched_output.csv"):
            return p
    except Exception:
        pass
    return None

def trigger_github_action():
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")           # "owner/repo"
    workflow_file = st.secrets.get("GITHUB_WORKFLOW_FILE", "nfl.yml")
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

# ==========================
# Odds helpers
# ==========================
def american_to_prob(o: float) -> float:
    try:
        o = float(o)
        return 100.0/(o+100.0) if o > 0 else abs(o)/(abs(o)+100.0)
    except Exception:
        return float("nan")

def prob_to_decimal(p: float) -> float:
    return 1.0/p if p and p > 0 else float("nan")

# ==========================
# Load per-book vig curves
# ==========================
def _default_vig_path() -> Path:
    # NFL/app_nfl_pp.py -> repo root -> models/vig_curves.json
    return Path(__file__).resolve().parents[1] / "models" / "vig_curves.json"

VIG_CURVES_PATH = os.getenv("VIG_CURVES_JSON", str(_default_vig_path()))
try:
    _VIG = json.loads(Path(VIG_CURVES_PATH).read_text())
except Exception:
    # fallback pooled (won’t be book-specific)
    _VIG = {"_pooled": {"a": 0.0, "b": 0.13, "c": 0.00033}}

def apply_vig_curve_per_book(o: float, book_name: str) -> float:
    """
    Fallback only: approximate fair odds for a single-sided quote using
    this book's vig curve (or pooled if unavailable). Applies 10% floor.
    """
    entry = _VIG.get(book_name, _VIG.get("_pooled", {"a":0.0,"b":0.13,"c":0.00033}))
    a, b, c = float(entry["a"]), float(entry["b"]), float(entry["c"])
    m = abs(o)
    vig = a + b*m + c*(m**2)
    floor = 0.10 * m
    vig = max(vig, floor, 0.0)
    return (m + vig) if o >= 0 else -(m - vig)

# ==========================
# Pair-first (no-vig) midpoint logic
# ==========================
OU_RE = re.compile(r"\b(Over|Under)\b", re.I)

def normalize_market(bt: str) -> str:
    s = str(bt).lower()
    if "total" in s or "over" in s or "under" in s: return "Total"
    if "money" in s: return "Moneyline"
    return "Total"  # NFL player props are generally O/U-style

def infer_side(row) -> str | None:
    m = OU_RE.search(str(row.get("Bet Type","")))
    if m: return m.group(1).title()
    sel = str(row.get("Selection","")).strip()
    if sel.lower() in ("over","under"): return sel.title()
    return None

def pair_midpoint_for_book(df: pd.DataFrame, book_col: str, event: str, market: str,
                           line_val, participant: str, side_label: str, offered: float) -> float | None:
    """
    Proper no-vig midpoint via probability normalization (handles -110/-110).
    For NFL player props (O/U): pair within (Event, Player, Market="Total", exact Line).
    """
    opp_side = None
    if side_label:
        s = side_label.lower()
        if s == "over": opp_side = "Under"
        elif s == "under": opp_side = "Over"

    pool = df[(df["Event"] == event) & (df["Market"] == market)]
    if "Player" in df.columns:
        pool = pool[pool["Player"] == participant]
    try:
        ln = float(line_val)
        pool = pool[pd.to_numeric(pool["Line"], errors="coerce") == ln]
    except Exception:
        pass

    opp_odds = None
    for _, r in pool.iterrows():
        r_side = str(r.get("_side","")).strip()
        if opp_side:
            if r_side != opp_side:
                continue
        else:
            if r_side.lower() == str(side_label or "").lower():
                continue
        v = r.get(book_col, None)
        if pd.notna(v) and str(v).strip() != "":
            try:
                opp_odds = float(v); break
            except Exception:
                pass

    if opp_odds is None:
        return None

    p1 = american_to_prob(offered)
    p2 = american_to_prob(opp_odds)
    denom = p1 + p2
    if not (math.isfinite(p1) and math.isfinite(p2)) or denom <= 0:
        return None
    p_true = p1 / denom

    if p_true == 0.5:
        fair = 100.0
    elif p_true < 0.5:
        fair = (100.0 / p_true) - 100.0
    else:
        fair = - (100.0 * p_true) / (1.0 - p_true)

    return fair if offered >= 0 else -abs(fair)

# ==========================
# Load data
# ==========================
def load_df():
    csv_path = _find_csv_path()
    if not csv_path or not csv_path.exists():
        st.error("nfl_matched_output.csv not found.")
        return None, None
    try:
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed')]
        df = df.dropna(axis=1, how="all")
        ts = datetime.fromtimestamp(csv_path.stat().st_mtime, pytz.utc)
        eastern = ts.astimezone(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %I:%M %p %Z")
        return df, eastern
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, None

# ==========================
# App
# ==========================
def run_app(df: pd.DataFrame | None = None):
    qp = st.query_params
    raw_mobile = qp.get("mobile", None)
    auto_mobile = "1" in raw_mobile if isinstance(raw_mobile, list) else raw_mobile == "1"

    if df is None:
        df, updated_ts = load_df()
        if df is None: return
        st.caption(f"Odds last updated: {updated_ts}")

    if df.empty:
        st.warning("No data to display."); return

    # Normalize schema
    df = df.rename(columns={
        "event": "Event",
        "game": "Event",
        "bet_type": "Bet Type",
        "player": "Player",
        "line": "Line",
        "selection": "Selection",
        "best_book": "Best Book",
        "best_odds": "Best Odds",
        "value_ratio": "Line vs. Average",
    })
    df["Market"] = df["Bet Type"].apply(normalize_market)

    fixed = {"Event","Bet Type","Market","Player","Line","Selection",
             "Best Book","Best Odds","Line vs. Average","kickoff_et"}
    book_cols = [c for c in df.columns if c not in fixed and not str(c).startswith("Unnamed")]

    def _valid(v): return pd.notna(v) and str(v).strip() != ""
    if book_cols:
        df["# Books"] = df[book_cols].apply(lambda r: sum(_valid(x) for x in r.values), axis=1)
    else:
        df["# Books"] = 0

    df["_side"] = df.apply(infer_side, axis=1)
    df["_line_num"] = pd.to_numeric(df.get("Line"), errors="coerce")

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        events = ["All"] + sorted(df["Event"].dropna().unique())
        sel_event = st.selectbox("Game", events, 0)
        players = ["All"] + sorted(df.get("Player", pd.Series(dtype=str)).dropna().unique())
        sel_player = st.selectbox("Player", players, 0)
        markets = ["All"] + sorted(df["Market"].dropna().unique())
        sel_market = st.selectbox("Market", markets, 0)
        books = ["All"] + sorted(df["Best Book"].dropna().unique()) if "Best Book" in df.columns else ["All"]
        sel_book = st.selectbox("Best Book", books, 0)
        min_books = st.number_input("Min. books posting this line", 1, 20, 2, 1)
        compact_mobile = st.toggle("Compact mobile mode", value=bool(auto_mobile))

    if sel_event != "All": df = df[df["Event"] == sel_event]
    if sel_player != "All" and "Player" in df.columns: df = df[df["Player"] == sel_player]
    if sel_market != "All": df = df[df["Market"] == sel_market]
    if sel_book != "All": df = df[df["Best Book"] == sel_book]
    df = df[df["# Books"] >= int(min_books)]

    # Percentify Line vs. Average
    df["Line vs. Average"] = pd.to_numeric(df.get("Line vs. Average"), errors="coerce")
    df["Line vs. Average (%)]"] = (df["Line vs. Average"] - 1.0) * 100.0

    # ==========================
    # Implied EV (%) with per-book curves
    # ==========================
    def fair_for_other_book(book: str, offered: float, row: pd.Series) -> float:
        fair = pair_midpoint_for_book(
            df=df,
            book_col=book,
            event=row.get("Event"),
            market=row.get("Market"),
            line_val=row.get("Line"),
            participant=row.get("Player"),
            side_label=row.get("_side"),
            offered=offered
        )
        if fair is not None:
            return fair
        return apply_vig_curve_per_book(offered, book)

    def calc_implied_ev(row: pd.Series) -> float:
        try:
            best_odds = float(row.get("Best Odds", float("nan")))
            if not math.isfinite(best_odds): return float("nan")
            best_book = row.get("Best Book")

            others = []
            for col in book_cols:
                if col == best_book:
                    continue
                v = row.get(col, None)
                try:
                    if pd.notna(v) and str(v).strip() != "":
                        others.append((col, float(v)))
                except Exception:
                    pass
            if not others: return float("nan")

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

    # ==========================
    # Formatting and display
    # ==========================
    def fmt_american_cell(x):
        try:
            x = int(float(x))
            return f"+{x}" if x > 0 else str(x)
        except Exception:
            return ""

    for col in book_cols:
        df[col] = df[col].apply(fmt_american_cell)
    if "Best Odds" in df.columns:
        df["Best Odds"] = df["Best Odds"].apply(fmt_american_cell)

    base_cols = ["Event","Player","Bet Type","Line","Selection"]
    is_mobile = bool(auto_mobile or compact_mobile)

    if sel_book != "All" and is_mobile:
        show_cols = [sel_book] if sel_book in book_cols else []
    else:
        show_cols = book_cols.copy()

    cols = base_cols + show_cols + ["Best Book","Line vs. Average (%)]","Implied EV (%)"]
    cols = [c for c in cols if c in df.columns]
    render_df = df[cols].copy()

    def _shorten(s, maxlen):
        s = str(s)
        return (s[:maxlen-1] + "…") if len(s) > maxlen else s

    if is_mobile:
        for c, maxlen in [("Event",22),("Player",18),("Selection",16)]:
            if c in render_df:
                render_df[c] = render_df[c].apply(lambda s: _shorten(s, maxlen))

    render_df = render_df.sort_values(by=["Implied EV (%)"], ascending=False, na_position="last")

    def _ev_green(val):
        try: v = float(val)
        except: return ""
        if not math.isfinite(v) or v <= 0: return ""
        cap = 20.0
        alpha = 0.15 + 0.8 * min(v/cap, 1.0)
        return f"background-color: rgba(34,139,34,{alpha}); font-weight:600;"

    def _shade_best_book(row):
        styles = [""] * len(row)
        bb = row.get("Best Book","")
        shade = _ev_green(row.get("Implied EV (%)", 0))
        if bb and bb in row.index:
            try:
                idx = list(row.index).index(bb)
                styles[idx] = shade
            except Exception:
                pass
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
        "Line vs. Average (%)]": "{:.1f}%",
        "Implied EV (%)": "{:.1f}%"
    })
    styled = styled.set_table_styles([{
        'selector':'th',
        'props':[('font-weight','bold'),
                 ('text-align','center'),
                 ('font-size','16px'),
                 ('background-color','#003366'),
                 ('color','white')]
    }])

    btn_cols = st.columns([1,1,1])
    with btn_cols[1]:
        if st.button("Refresh Odds", use_container_width=True):
            if trigger_github_action():
                wait_for_csv_update()

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

if __name__ == "__main__":
    run_app()
