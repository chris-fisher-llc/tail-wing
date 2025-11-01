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

# --- Make sidebar (filters) narrower (~half of default) ---
st.markdown(
    """
    <style>
      /* Narrow the sidebar */
      section[data-testid="stSidebar"] { width: 12rem !important; min-width: 12rem !important; }
      div[data-testid="stSidebar"] > div { width: 12rem !important; }

      /* Keep widgets readable in narrower sidebar */
      section[data-testid="stSidebar"] * { font-size: 0.87rem !important; }
      section[data-testid="stSidebar"] label { white-space: normal !important; }
    </style>
    """,
    unsafe_allow_html=True
)

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

# ---- GitHub Action trigger ----
def trigger_github_action():
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO")          # e.g. "chris-fisher-llc/tail-wing"
    workflow_file = st.secrets.get("GITHUB_WORKFLOW_FILE", "cfb.yml")
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
    env = os.getenv("cfb_matched_output_CSV")
    if env:
        p = Path(env)
        if p.exists(): return p
    here = Path(__file__).resolve().parent
    candidates = [
        here / "cfb_matched_output.csv",
        here / "cfb" / "cfb_matched_output.csv",
        here.parent / "cfb_matched_output.csv",
        here.parent / "cfb" / "cfb_matched_output.csv",
        Path.cwd() / "cfb_matched_output.csv",
        Path.cwd() / "cfb" / "cfb_matched_output.csv",
    ]
    for p in candidates:
        if p.exists(): return p
    try:
        for p in here.rglob("cfb_matched_output.csv"):
            return p
    except Exception:
        pass
    return None

# ---------- Helpers ----------
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
    bt = str(row.get("Bet Type",""))
    m = OU_RE.search(bt)
    if m:
        return m.group(1).title()
    for key in ("selection","Selection","team","Team"):
        if key in row.index and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
    return None

def to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None

def round_to_half(x: float | None) -> float | None:
    if x is None: return None
    return round(x * 2) / 2

# ---- Name normalization for sportsbook matching (fixes Best Book filter/shading) ----
def _normkey(s) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def pair_midpoint_for_book(df: pd.DataFrame, book_col: str, event: str, market: str,
                           line_val, side_label: str, offered: float) -> float | None:
    def opponent_of(side: str) -> str | None:
        if market == "Total":
            return "Under" if side and side.lower()=="over" else ("Over" if side and side.lower()=="under" else None)
        if " @ " in str(event):
            away, home = [x.strip() for x in str(event).split(" @ ", 1)]
            if side and side.lower()==away.lower(): return home
            if side and side.lower()==home.lower(): return away
        return None

    opp_side = opponent_of(side_label)
    pool = df[(df["Event"]==event) & (df["Market"]==market)]
    if market == "Total":
        try:
            ln = float(line_val)
            pool = pool[pd.to_numeric(pool["Line"], errors="coerce") == ln]
        except Exception:
            pass
    elif market == "Spread":
        try:
            ln = round(float(line_val) * 2) / 2
            pool = pool[pool["|Line|"] == abs(ln) if ln is not None else False]
        except Exception:
            pass

    opp_odds = None
    for _, r in pool.iterrows():
        r_side = str(r.get("_side","")).strip()
        if opp_side:
            if r_side.lower() != opp_side.lower():
                continue
        else:
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

    def to_prob(o: float):
        return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

    p1, p2 = to_prob(offered), to_prob(opp_odds)
    if not (math.isfinite(p1) and math.isfinite(p2)) or (p1+p2)==0:
        return None

    p_true = p1 / (p1 + p2)

    if p_true == 0.5:
        fair = 100.0
    elif p_true < 0.5:
        fair = (100 / p_true) - 100
    else:
        fair = - (100 * p_true) / (1 - p_true)

    return fair if offered >= 0 else -abs(fair)

# ---------- App core ----------
def run_app(df: pd.DataFrame | None = None):
    # Mobile flag
    qp = st.query_params
    raw_mobile = qp.get("mobile", None)
    auto_mobile = "1" in raw_mobile if isinstance(raw_mobile, list) else raw_mobile == "1"

    # Load CSV
    if df is None:
        csv_path = _find_csv_path()
        if not csv_path or not csv_path.exists():
            st.error("cfb_matched_output.csv not found.")
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
        "line":"Line",
        "selection":"Selection",
        "opponent":"Opponent",
        "best_book":"Best Book",
        "best_odds":"Best Odds",
        "value_ratio":"Line vs. Average",
        "best_decimal":"best_decimal",
        "avg_other_decimal":"avg_other_decimal",
        "value_flag":"value_flag",
    })

    # Rename sportsbook columns: drop trailing "_Odds"
    new_cols = {}
    for c in df.columns:
        if c.endswith("_Odds"):
            new_cols[c] = c[:-5]
    df = df.rename(columns=new_cols)

    # Clean Best Book values
    if "Best Book" in df.columns:
        df["Best Book"] = df["Best Book"].astype(str).str.replace("_Odds","", regex=False).str.strip()

    # Compute Market normalization
    df["Market"] = df["Bet Type"].apply(normalize_market)

    # Dynamic sportsbook columns
    fixed = {"Event","Bet Type","Market","Line","Selection","Opponent",
             "Best Book","Best Odds","Line vs. Average","kickoff_et",
             "best_decimal","avg_other_decimal","value_flag"}
    book_cols = [c for c in df.columns if c not in fixed and not str(c).startswith("Unnamed")]

    # ---- Build mapping between various book name spellings and column names ----
    book_map = { _normkey(c): c for c in book_cols }           # normalized key -> exact column name
    if "Best Book" in df.columns:
        df["Best Book Key"] = df["Best Book"].apply(_normkey)  # normalize CSV values

    # Count #books
    def _is_valid_num(v): return pd.notnull(v) and str(v).strip() != ""
    if book_cols:
        df["# Books"] = df[book_cols].apply(lambda r: sum(_is_valid_num(x) for x in r.values), axis=1)
    else:
        df["# Books"] = 0

    # Sidebar (books derived from actual columns to avoid case/spacing mismatches)
    with st.sidebar:
        st.header("Filters")
        evs = ["All"] + sorted(df["Event"].dropna().unique())
        sel_event = st.selectbox("Game", evs, 0)
        markets = ["All"] + sorted(df["Market"].dropna().unique())
        sel_market = st.selectbox("Market", markets, 0)
        books = ["All"] + sorted(book_cols)    # use column names (pretty) instead of raw CSV values
        sel_book = st.selectbox("Best Book", books, 0)
        min_books = st.number_input("Min. books posting this line", 1, 20, 2, 1)
        compact_mobile = st.toggle("Compact mobile mode", value=bool(auto_mobile))

    # Apply filters
    if sel_event!="All": df=df[df["Event"]==sel_event]
    if sel_market!="All": df=df[df["Market"]==sel_market]
    if sel_book!="All":
        # Match rows whose normalized Best Book equals the normalized selected column label
        df = df[df.get("Best Book Key","") == _normkey(sel_book)]
    df = df[df["# Books"]>=int(min_books)]

    # Percentify "Line vs. Average"
    df["Line vs. Average"] = pd.to_numeric(df.get("Line vs. Average"), errors="coerce")
    df["Line vs. Average (%)"] = (df["Line vs. Average"] - 1.0) * 100.0

    # Side inference and keys for pairing
    df["_side"] = df.apply(infer_side, axis=1)

    # Pre-compute numeric line (for pairing & formatting)
    df["_line_num"] = pd.to_numeric(df.get("Line"), errors="coerce")

    def _abs_key(row):
        try:
            if row["Market"] != "Spread":
                return None
            ln = round_to_half(to_float_or_none(row["_line_num"]))
            return abs(ln) if ln is not None else None
        except Exception:
            return None
    df["|Line|"] = df.apply(_abs_key, axis=1)

    # ---------- Implied EV: pair-first, curve-fallback ----------
    def fair_for_other_book(book: str, offered: float, row: pd.Series) -> float:
        fair = pair_midpoint_for_book(
            df=df,
            book_col=book,
            event=row.get("Event"),
            market=row.get("Market"),
            line_val=row.get("Line"),
            side_label=row.get("_side"),
            offered=offered
        )
        if fair is not None:
            return fair
        return apply_vig_curve(offered)

    def calc_implied_ev(row: pd.Series) -> float:
        try:
            best_odds = float(row.get("Best Odds", float("nan")))
            if not math.isfinite(best_odds): return float("nan")

            # Gather "other" books that post a price
            others = []
            for col in book_cols:
                if col == book_map.get(row.get("Best Book Key",""), None):
                    continue
                v = row.get(col, None)
                try:
                    if pd.notna(v) and str(v).strip()!="":
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

    # ---------- Pretty-format AFTER EV is computed ----------
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

    # Format Line
    def fmt_line(val, market):
        if market == "Moneyline" or pd.isna(val):
            return ""
        try:
            x = float(val)
            if abs(x*2 - round(x*2)) < 1e-9:
                if abs(x*2) % 2 == 1:
                    return f"{x:.1f}"
                else:
                    return f"{int(x)}"
            xh = round_to_half(x)
            return f"{xh:.1f}" if abs(xh*2) % 2 == 1 else f"{int(xh)}"
        except Exception:
            return str(val) if pd.notna(val) else ""

    df["Line"] = [fmt_line(v, m) for v, m in zip(df.get("_line_num"), df.get("Market"))]

    # ---------- Build render dataframe ----------
    base_cols = ["Event","Bet Type","Line","Selection","Opponent"]
    is_mobile = bool(auto_mobile or compact_mobile)

    if sel_book!="All" and is_mobile:
        show_cols = [sel_book] if sel_book in book_cols else []
    else:
        show_cols = book_cols.copy()

    # Hide requested + internals (Best Book column is now hidden)
    hidden = {"# Books", "Best Book", "Best Odds", "Opponent", "best_decimal",
              "avg_other_decimal", "value_flag", "_line_num", "|Line|", "_side", "Best Book Key"}
    cols = base_cols + show_cols + ["Line vs. Average (%)","Implied EV (%)"]
    cols = [c for c in cols if c in df.columns and c not in hidden]
    render_df = df[cols].copy()

    # Compact cosmetics
    def _shorten(s,maxlen):
        s=str(s)
        return (s[:maxlen-1]+"…") if len(s)>maxlen else s
    if is_mobile:
        if "Event" in render_df:     render_df["Event"]    = render_df["Event"].apply(lambda s:_shorten(s,22))
        if "Selection" in render_df: render_df["Selection"]=render_df["Selection"].apply(lambda s:_shorten(s,18))
        if "Opponent" in render_df:  render_df["Opponent"] = render_df["Opponent"].apply(lambda s:_shorten(s,18))

    render_df = render_df.sort_values(by=["Implied EV (%)"], ascending=False, na_position="last")

    # ---------- Styling ----------
    def _ev_green(val):
        try: v=float(val)
        except: return ""
        if not math.isfinite(v) or v<=0: return ""
        cap=20.0; alpha=0.15+0.8*min(v/cap,1.0)
        return f"background-color: rgba(34,139,34,{alpha}); font-weight:600;"

    # Shade the winning sportsbook cell using normalized name mapping
    def _shade_best_book(row):
        styles=[""]*len(row)
        # We need the original df row to get Best Book Key; rebuild key via display row if needed
        # Here we can’t access the hidden columns, so we compute from available columns:
        # Instead we rely on the fact that render_df is aligned with df (same order), so we’ll
        # attach a computed function after creating styled DataFrame (simpler: map within df then join).
        return styles

    # Create style on full render_df, plus a parallel list of best-book target columns
    # Build a vector with the actual column name to shade, derived from df's Best Book Key
    bb_target_col = df.loc[render_df.index, "Best Book Key"].map(lambda k: book_map.get(k, None)) if "Best Book Key" in df.columns else pd.Series([None]*len(render_df), index=render_df.index)

    # style function that can see bb_target_col via closure
    def _shade_row(row):
        styles=[""]*len(row)
        shade=_ev_green(row.get("Implied EV (%)",0))
        target = bb_target_col.loc[row.name]
        if target and target in row.index:
            try:
                idx=list(row.index).index(target)
                styles[idx]=shade
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
    styled = styled.apply(_shade_row, axis=1)
    styled = styled.format({
        "Line vs. Average (%)": "{:.1f}%",
        "Implied EV (%)": "{:.1f}%"
    })
    styled = styled.set_table_styles([{
        'selector':'th','props':[('font-weight','bold'),('text-align','center'),
        ('font-size','16px'),('background-color','#003366'),('color','white')]
    }])

    # ---- Centered refresh button ----
    btn_cols = st.columns([1,1,1])
    with btn_cols[1]:
        if st.button("Refresh Odds", use_container_width=True):
            if trigger_github_action():
                wait_for_csv_update()

    st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)

# ---- Run ----
if __name__=="__main__":
    run_app()
