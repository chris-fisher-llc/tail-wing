###############################################################
#   CLEAN CFB STREAMLIT BOARD  —  SPREADS • TOTALS • MONEYLINES
#   Matches new pull script 1:1
#   Includes correct EV, book filtering, pair-first fair odds,
#   and dynamic sportsbook detection.
###############################################################

import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
from pathlib import Path
import pytz
import os

# -------------------------------------------------------------
# Page setup
# -------------------------------------------------------------
st.set_page_config(page_title="College Football — Game Lines", layout="wide")

# Wider sidebar
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    width: 14rem !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Utility: Locate the CSV from pull script
# -------------------------------------------------------------
def find_csv():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "cfb_matched_output.csv",
        here.parent / "cfb_matched_output.csv",
        Path.cwd() / "cfb_matched_output.csv"
    ]
    for c in candidates:
        if c.exists():
            return c
    for p in here.rglob("cfb_matched_output.csv"):
        return p
    return None


# -------------------------------------------------------------
# Odds conversions
# -------------------------------------------------------------
def american_to_prob(o):
    try:
        o = float(o)
    except:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def prob_to_decimal(p):
    if not p or p <= 0:
        return None
    return 1.0 / p


def prob_to_american(p):
    if p <= 0 or p >= 1:
        return None
    if p < 0.5:
        return round(100 * (1 - p) / p)
    return round(-100 * p / (1 - p))


# Vig curve fallback (same coefficients as NFL/NBA)
def vigcurve_fair_prob(odds):
    try:
        o = float(odds)
    except:
        return None
    p = american_to_prob(o)
    if p is None:
        return None

    base = 0.045
    decay = min(1.0, 100 / (100 + abs(o)))
    margin = base * decay

    p2 = p * (1 - margin)
    return max(min(p2, 0.999), 0.001)


# -------------------------------------------------------------
# Pair-first fair odds
# -------------------------------------------------------------
def compute_pair_first_fair(df_event, team, opp, market, line):
    team_rows = df_event[df_event["_side"] == team]
    opp_rows  = df_event[df_event["_side"] == opp]

    if team_rows.empty or opp_rows.empty:
        return None

    pair_probs = []

    for _, trow in team_rows.iterrows():
        for _, orow in opp_rows.iterrows():

            # Match lines for spreads/totals
            if market != "Moneyline":
                if pd.isna(trow["_line_num"]) or pd.isna(orow["_line_num"]):
                    continue
                if abs(trow["_line_num"] - orow["_line_num"]) > 0.01:
                    continue

            t_odds = trow["Best Odds Raw"]
            o_odds = orow["Best Odds Raw"]
            if t_odds is None or o_odds is None:
                continue

            p_t = american_to_prob(t_odds)
            p_o = american_to_prob(o_odds)
            if not p_t or not p_o:
                continue

            fair = p_t / (p_t + p_o)
            pair_probs.append(fair)

    if not pair_probs:
        return None

    return float(np.mean(pair_probs))


# -------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------
csv_path = find_csv()
if not csv_path:
    st.error("cfb_matched_output.csv not found.")
    st.stop()

df = pd.read_csv(csv_path)

# Timestamp
ts = datetime.fromtimestamp(csv_path.stat().st_mtime, pytz.utc)
est = ts.astimezone(pytz.timezone("US/Eastern")).strftime("%Y-%m-%d %I:%M %p %Z")
st.caption(f"Odds last updated: {est}")

# Normalize column names
df = df.rename(columns={
    "game": "Event",
    "bet_type": "Bet Type",
    "selection": "Selection",
    "opponent": "Opponent",
    "line": "Line",
    "best_book": "Best Book",
    "best_odds": "Best Odds Raw",
})

df["Market"] = df["Bet Type"].apply(
    lambda x: "Moneyline" if "money" in x.lower()
              else ("Total" if "over" in x.lower() or "under" in x.lower() or "total" in x.lower()
              else "Spread")
)

df["_side"] = df["Selection"]
df["_line_num"] = pd.to_numeric(df["Line"], errors="coerce")

# -------------------------------------------------------------
# Detect sportsbook columns dynamically
# -------------------------------------------------------------
EXCLUDE = {"BetOnlineAG", "BetUS", "LowVig", "MyBookieAG"}

fixed = {"Event", "kickoff_et", "Bet Type", "Selection", "Opponent", "Line",
         "Market", "Best Book", "Best Odds Raw", "_side", "_line_num"}

book_cols = [c for c in df.columns if c not in fixed and c not in EXCLUDE]

# -------------------------------------------------------------
# Compute Best Book from row data
# -------------------------------------------------------------
def compute_best(r):
    best_book = None
    best_odds = None
    best_dec = -1

    for bk in book_cols:
        v = r.get(bk)
        try:
            v = float(v)
        except:
            continue
        dec = prob_to_decimal(american_to_prob(v))
        if dec and dec > best_dec:
            best_dec = dec
            best_book = bk
            best_odds = v
    return pd.Series({"BestBookCol": best_book, "Best Odds Raw": best_odds})

bb = df.apply(compute_best, axis=1)
df["BestBookCol"] = bb["BestBookCol"]
df["Best Odds Raw"] = bb["Best Odds Raw"]

# -------------------------------------------------------------
# EV calculation
# -------------------------------------------------------------
def calc_ev(r):
    best_odds = r["Best Odds Raw"]
    if best_odds is None or not np.isfinite(best_odds):
        return np.nan

    event = r["Event"]
    side  = r["_side"]
    opp   = r["Opponent"]
    market = r["Market"]
    line   = r["_line_num"]

    df_event = df[df["Event"] == event]

    fair_p = compute_pair_first_fair(df_event, side, opp, market, line)
    if fair_p is None:
        fair_p = vigcurve_fair_prob(best_odds)

    fair_dec = prob_to_decimal(fair_p)
    best_dec = prob_to_decimal(american_to_prob(best_odds))

    if fair_dec and best_dec:
        return (best_dec / fair_dec - 1) * 100

    return np.nan

df["Implied EV (%)"] = df.apply(calc_ev, axis=1)

# -------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    events = ["All"] + sorted(df["Event"].unique())
    sel_event = st.selectbox("Game", events, 0)

    markets = ["All"] + sorted(df["Market"].unique())
    sel_market = st.selectbox("Market", markets, 0)

    books = ["All"] + book_cols
    sel_book = st.selectbox("Best Book", books, 0)

    max_books = max(1, len(book_cols))
    default_val = min(2, max_books)
    min_books = st.number_input("Min. books posting this line",
                                min_value=1, max_value=max_books,
                                value=default_val, step=1)

# -------------------------------------------------------------
# Apply filters
# -------------------------------------------------------------
if sel_event != "All":
    df = df[df["Event"] == sel_event]
if sel_market != "All":
    df = df[df["Market"] == sel_market]
if sel_book != "All":
    df = df[df["BestBookCol"] == sel_book]

# Book count
df["#Books"] = df[book_cols].apply(lambda r: sum(str(x).strip() != "" for x in r), axis=1)
df = df[df["#Books"] >= int(min_books)]

# -------------------------------------------------------------
# Final formatting
# -------------------------------------------------------------
def fmt_amer(x):
    try:
        x = int(float(x))
        return f"+{x}" if x > 0 else str(x)
    except:
        return ""

for bk in book_cols:
    df[bk] = df[bk].apply(fmt_amer)

df["Best Odds"] = df["Best Odds Raw"].apply(fmt_amer)
df["Line vs. Average (%)"] = (pd.to_numeric(df["avg_other_decimal"], errors="coerce") - 1) * 100

# Build output table
show = ["Event", "Bet Type", "Line", "Selection"] + book_cols + ["Implied EV (%)"]
out = df[show].sort_values("Implied EV (%)", ascending=False)

st.dataframe(out, use_container_width=True)
