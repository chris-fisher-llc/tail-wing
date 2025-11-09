import os, json, math, re, time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="The Tail Wing - NFL Player Props", layout="wide")

# ---------------------------
# Config / Secrets
# ---------------------------
BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"
DEFAULT_REGIONS = "us,us2"

# Markets: feel free to tweak; these are common O/U-style NFL player props + a Yes/No market.
DEFAULT_MARKETS = [
    "player_passing_yards",
    "player_rushing_yards",
    "player_receiving_yards",
    "player_receptions",
    "player_passing_touchdowns",
    "player_rushing_touchdowns",
    "player_interceptions",
    "player_anytime_touchdown_scorer"  # Yes/No
]

def _get_api_key():
    for k in ("ODDS_API_KEY", "THE_ODDS_API_KEY"):
        if k in st.secrets and st.secrets[k]:
            return st.secrets[k]
    st.error("Missing API key in st.secrets (use ODDS_API_KEY or THE_ODDS_API_KEY).")
    st.stop()

API_KEY = _get_api_key()

# ---------------------------
# Per-book vig curves
# ---------------------------
def _default_vig_path() -> Path:
    # NFL/app_nfl_pp.py -> repo root -> models/vig_curves.json
    return Path(__file__).resolve().parents[1] / "models" / "vig_curves.json"

VIG_CURVES_PATH = os.getenv("VIG_CURVES_JSON", str(_default_vig_path()))
try:
    _VIG = json.loads(Path(VIG_CURVES_PATH).read_text())
except Exception:
    _VIG = {"_pooled": {"a": 0.0, "b": 0.13, "c": 0.00033}}

# ---------------------------
# Helpers
# ---------------------------
def american_to_prob(o: float) -> float:
    o = float(o)
    return 100.0/(o+100.0) if o > 0 else abs(o)/(abs(o)+100.0)

def prob_to_decimal(p: float) -> float:
    return 1.0/p if (p and p > 0) else float("nan")

def american_to_decimal(o: float) -> float:
    o = float(o)
    return 1.0 + (o/100.0) if o > 0 else 1.0 + (100.0/abs(o))

def apply_vig_curve_per_book(o: float, book_name: str) -> float:
    """
    Fallback only: approximate fair odds for a single-sided quote using this book's curve (or pooled).
    Applies 10% floor of magnitude.
    """
    entry = _VIG.get(book_name, _VIG.get("_pooled", {"a":0.0,"b":0.13,"c":0.00033}))
    a, b, c = float(entry["a"]), float(entry["b"]), float(entry["c"])
    m = abs(o)
    vig = a + b*m + c*(m**2)
    floor = 0.10 * m
    vig = max(vig, floor, 0.0)
    return (m + vig) if o >= 0 else -(m - vig)

OU_RE = re.compile(r"\b(Over|Under)\b", re.I)

def normalize_market_key(market_key: str) -> str:
    # User-friendly label
    mk = (market_key or "").replace("_", " ").title()
    return mk

def selection_from_outcome_name(name: str) -> str:
    s = (name or "").strip()
    if s.lower() in ("over","under","yes","no"):
        return s.title()
    return s

def compute_pair_midpoint(offered: float, opp: float) -> float:
    """No-vig midpoint via probability normalization (handles -110/-110). Returns fair American for 'offered' side."""
    p1 = american_to_prob(offered)
    p2 = american_to_prob(opp)
    denom = p1 + p2
    if denom <= 0:
        return None
    p_true = p1 / denom
    # back to American
    if p_true == 0.5:
        fair = 100.0
    elif p_true < 0.5:
        fair = (100.0 / p_true) - 100.0
    else:
        fair = - (100.0 * p_true) / (1.0 - p_true)
    return fair if offered >= 0 else -abs(fair)

def best_book_for_row(row: pd.Series, book_cols: list[str]) -> tuple[str, float]:
    """Pick bettor-favorable price: max decimal payout."""
    best_b, best_o, best_dec = None, None, -1.0
    for b in book_cols:
        v = row.get(b, None)
        if v is None or (isinstance(v, float) and not math.isfinite(v)): 
            continue
        try:
            o = float(v)
            d = american_to_decimal(o)
            if d > best_dec:
                best_dec, best_b, best_o = d, b, o
        except:
            pass
    return best_b, best_o

# ---------------------------
# The Odds API client
# ---------------------------
def fetch_player_props(markets: list[str], regions: str = DEFAULT_REGIONS, days_ahead: int = 2) -> list[dict]:
    """
    Pull upcoming NFL events with requested markets.
    """
    url = f"{BASE}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Odds API error {r.status_code}: {r.text}")
        st.stop()
    return r.json()

def flatten_to_rows(events_json: list[dict]) -> pd.DataFrame:
    """
    Build wide table: one row per (Event, Player/Participant, Market, Line, Selection),
    columns per-book with the offered American odds.
    """
    rows = []
    for ev in events_json:
        event = f"{ev.get('away_team','')} @ {ev.get('home_team','')}".strip()
        commence = ev.get("commence_time")
        for bk in ev.get("bookmakers", []):
            book = bk.get("title") or bk.get("key")
            for m in bk.get("markets", []):
                mkey = m.get("key") or ""
                market_label = normalize_market_key(mkey)
                for out in m.get("outcomes", []):
                    sel = selection_from_outcome_name(out.get("name"))
                    odds = out.get("price")
                    # Player/participant & line where available
                    player = out.get("description") or out.get("player") or ""
                    line = out.get("point")
                    # Only keep markets that are essentially two-sided (O/U or Yes/No or team-side props)
                    if sel not in ("Over","Under","Yes","No"):
                        # still include (e.g., team names), but pairing will rely on opposite name
                        pass
                    rows.append({
                        "Event": event,
                        "kickoff_et": commence,
                        "Player": player,
                        "Bet Type": market_label,
                        "MarketKey": mkey,
                        "Line": line,
                        "Selection": sel,
                        "Book": book,
                        "Odds": odds
                    })
    if not rows:
        return pd.DataFrame()

    long_df = pd.DataFrame(rows)

    # Build a composite key to pivot on
    # For O/U props we need same Event+Player+Market+Line+Selection
    long_df["Line"] = pd.to_numeric(long_df["Line"], errors="coerce")
    key_cols = ["Event","Player","Bet Type","MarketKey","Line","Selection"]

    # Pivot to wide: one column per book, values = Odds
    wide = long_df.pivot_table(index=key_cols, columns="Book", values="Odds", aggfunc="first").reset_index()

    # Keep a stable column order: identifiers first, then books alpha
    id_cols = key_cols.copy()
    book_cols = sorted([c for c in wide.columns if c not in id_cols])
    wide = wide[id_cols + book_cols]
    return wide

# ---------------------------
# Implied EV computation
# ---------------------------
def pair_midpoint_from_row_rowbook(wide_df: pd.DataFrame, row: pd.Series, book_col: str) -> float | None:
    """Find the opposite side row for the same Event/Player/Market/Line and compute fair via probability normalization."""
    event, player, mkt, mkey, line, sel = (row["Event"], row["Player"], row["Bet Type"], row["MarketKey"], row["Line"], row["Selection"])
    # determine opposite selection for common binary markets
    opp_sel = None
    s = str(sel).lower()
    if s == "over": opp_sel = "Under"
    elif s == "under": opp_sel = "Over"
    elif s == "yes": opp_sel = "No"
    elif s == "no": opp_sel = "Yes"
    else:
        # try no pairing for non-binary labels (team vs team props would need richer logic)
        return None

    mask = (
        (wide_df["Event"]==event) &
        (wide_df["Player"]==player) &
        (wide_df["Bet Type"]==mkt) &
        (wide_df["MarketKey"]==mkey) &
        (pd.to_numeric(wide_df["Line"], errors="coerce")==pd.to_numeric(line)) &
        (wide_df["Selection"]==opp_sel)
    )
    opp_rows = wide_df.loc[mask]
    if opp_rows.empty: 
        return None
    try:
        offered = float(row[book_col])
        opp = float(opp_rows.iloc[0][book_col])
    except Exception:
        return None
    if not (math.isfinite(offered) and math.isfinite(opp)):
        return None
    return compute_pair_midpoint(offered, opp)

def compute_implied_ev_for_board(wide: pd.DataFrame) -> pd.DataFrame:
    """Add # Books, Best Book, Best Odds, Line vs. Average (%), and Implied EV (%)."""
    # Identify sportsbook columns
    fixed = {"Event","Player","Bet Type","MarketKey","Line","Selection"}
    book_cols = [c for c in wide.columns if c not in fixed]

    # #Books posting this exact row
    def _valid(v): 
        return pd.notna(v) and str(v).strip() != ""
    wide["# Books"] = wide[book_cols].apply(lambda r: sum(_valid(x) for x in r.values), axis=1)

    # Best book & odds
    bb_data = wide.apply(lambda r: best_book_for_row(r, book_cols), axis=1, result_type="expand")
    wide["Best Book"] = bb_data[0]
    wide["Best Odds"] = bb_data[1]

    # Line vs. Average (%) — legacy visual (decimal ratio - 1)
    def line_vs_avg(row):
        vals = []
        for b in book_cols:
            v = row.get(b, None)
            if pd.notna(v) and str(v).strip()!="":
                try:
                    vals.append(american_to_decimal(float(v)))
                except:
                    pass
        if not vals:
            return float("nan")
        best_dec = american_to_decimal(float(row.get("Best Odds")))
        avg_dec = sum(vals)/len(vals)
        return best_dec/avg_dec
    wide["Line vs. Average"] = wide.apply(line_vs_avg, axis=1)
    wide["Line vs. Average (%)]"] = (wide["Line vs. Average"] - 1.0) * 100.0

    # Implied EV (%)
    def implied_ev_row(row: pd.Series) -> float:
        try:
            best_odds = float(row.get("Best Odds", float("nan")))
            best_book = row.get("Best Book")
            if not best_book or not math.isfinite(best_odds):
                return float("nan")

            # Build fair probs from other books
            fair_probs = []
            for b in book_cols:
                if b == best_book: 
                    continue
                v = row.get(b, None)
                if v is None or (isinstance(v, float) and not math.isfinite(v)): 
                    continue
                try:
                    offered = float(v)
                except:
                    continue

                # Pair-first within same book if possible
                fair_american = pair_midpoint_from_row_rowbook(wide, row, b)
                if fair_american is None:
                    # Fallback: per-book vig curve
                    fair_american = apply_vig_curve_per_book(offered, b)

                p = american_to_prob(fair_american)
                if math.isfinite(p) and p > 0:
                    fair_probs.append(p)

            if not fair_probs:
                return float("nan")

            avg_prob = sum(fair_probs) / len(fair_probs)
            true_decimal = prob_to_decimal(avg_prob)
            best_dec = american_to_decimal(best_odds)
            if not (math.isfinite(true_decimal) and math.isfinite(best_dec)):
                return float("nan")
            return (best_dec / true_decimal - 1.0) * 100.0
        except Exception:
            return float("nan")

    wide["Implied EV (%)"] = wide.apply(implied_ev_row, axis=1)
    return wide

# ---------------------------
# UI bootstrap (mobile)
# ---------------------------
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
    st.session_state.setdefault("_awaited_mobile_param_nfl_live", False)
    if not st.session_state["_awaited_mobile_param_nfl_live"]:
        st.session_state["_awaited_mobile_param_nfl_live"] = True
        st.autorefresh(interval=250, limit=1, key="await_mobile_param_nfl_live")
except Exception:
    pass

# ---------------------------
# App
# ---------------------------
st.markdown("""
<h1 style='text-align:center; font-size:42px;'>NFL — Player Props</h1>
<p style='text-align:center; color:gray; font-size:18px;'>
Live odds → Pair-first fair pricing → Per-book vig curves (fallback)
</p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Fetch Settings")
    regions = st.text_input("Regions", value=DEFAULT_REGIONS, help="Comma-separated (e.g., us,us2)")
    markets = st.text_area("Markets (comma-separated)", value=",".join(DEFAULT_MARKETS))
    st.caption("Tip: keep to O/U style markets for best pairing; Yes/No also supported.")

    st.header("Display Filters")
    compact_mobile = st.toggle("Compact mobile mode", value=("1" in (st.query_params.get("mobile", ["0"]))))

# Fetch button
fetch_col = st.columns([1,1,1])[1]
with fetch_col:
    if st.button("Refresh Odds", use_container_width=True):
        st.experimental_rerun()

# Always fetch on load/rerun (idempotent enough)
with st.spinner("Pulling live NFL player props…"):
    events = fetch_player_props([m.strip() for m in markets.split(",") if m.strip()], regions.strip() or DEFAULT_REGIONS)
    wide = flatten_to_rows(events)

if wide.empty:
    st.warning("No odds returned. Try adjusting markets/regions or retry shortly.")
    st.stop()

# Compute board metrics (Best Book, Line vs Avg, Implied EV)
wide = compute_implied_ev_for_board(wide)

# Sidebar filters for the resulting table
with st.sidebar:
    events_list = ["All"] + sorted(wide["Event"].dropna().unique().tolist())
    players_list = ["All"] + sorted(wide["Player"].dropna().unique().tolist())
    markets_list = ["All"] + sorted(wide["Bet Type"].dropna().unique().tolist())
    bestbooks_list = ["All"] + sorted(wide["Best Book"].dropna().unique().tolist())
    min_books = st.number_input("Min. books posting this line", 1, 20, 2, 1)

    sel_event = st.selectbox("Game", events_list, 0)
    sel_player = st.selectbox("Player", players_list, 0)
    sel_market = st.selectbox("Market", markets_list, 0)
    sel_bestbook = st.selectbox("Best Book", bestbooks_list, 0)

# Apply filters
df = wide.copy()
if sel_event != "All": df = df[df["Event"]==sel_event]
if sel_player != "All": df = df[df["Player"]==sel_player]
if sel_market != "All": df = df[df["Bet Type"]==sel_market]
if sel_bestbook != "All": df = df[df["Best Book"]==sel_bestbook]
df = df[df["# Books"] >= int(min_books)]

# Build column set for display
fixed = {"Event","Player","Bet Type","MarketKey","Line","Selection","Best Book","Best Odds","# Books",
         "Line vs. Average","Line vs. Average (%)]","Implied EV (%)"}
book_cols = [c for c in df.columns if c not in fixed]
base_cols = ["Event","Player","Bet Type","Line","Selection"]

# “Best cell” highlight uses the raw book columns; format them to +/-
def fmt_american_cell(x):
    try:
        x = float(x)
        xi = int(x)
        return f"+{xi}" if xi > 0 else str(xi)
    except:
        return ""

for c in book_cols:
    df[c] = df[c].apply(fmt_american_cell)
df["Best Odds"] = df["Best Odds"].apply(fmt_american_cell)

# Mobile sizing tweaks
is_mobile = bool(compact_mobile)
def _shorten(s, maxlen):
    s = str(s)
    return (s[:maxlen-1]+"…") if len(s)>maxlen else s

if is_mobile:
    for c, maxlen in [("Event",22),("Player",18),("Selection",16)]:
        if c in df.columns:
            df[c] = df[c].apply(lambda s: _shorten(s, maxlen))

# Assemble final render
show_cols = base_cols + book_cols + ["Best Book","Implied EV (%)","Line vs. Average (%)]","# Books"]
df = df[show_cols].copy()
df = df.sort_values(by=["Implied EV (%)"], ascending=False, na_position="last")

# Styling: shade Implied EV green and also shade the Best Book cell in that row
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
        except:
            pass
    return styles

st.markdown("""
<style>
  [data-testid="stDataFrame"] * {font-size:0.92rem;}
</style>
""", unsafe_allow_html=True)

styled = df.style
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

st.dataframe(styled, use_container_width=True, hide_index=True, height=1200)
