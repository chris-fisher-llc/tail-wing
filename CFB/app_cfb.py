# =======================================================================
#   CFB GAME LINES — CLEAN, FIXED, DROP-IN REPLACEMENT
#   Includes:
#     • Correct Best Book filter (works for moneylines)
#     • Correct pair-first EV calculation
#     • Excluded books removed: BetOnlineAG, BetUS, LowVig, MyBookieAG
#     • WilliamHillUS → Caesars
#     • Added ESPNBet + HardRock (if present)
#     • Sidebar styling, +/- buttons
# =======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz
import math
import time
from pathlib import Path

# -------------------------------------------------------------
# Page config + Sidebar width
# -------------------------------------------------------------
st.set_page_config(page_title="College Football — Game Lines", layout="wide")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { width: 14rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# Utility locators
# -------------------------------------------------------------
def _find_csv() -> Path | None:
    """Locate cfb_matched_output.csv in common locations."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "cfb_matched_output.csv",
        here.parent / "cfb_matched_output.csv",
        Path.cwd() / "cfb_matched_output.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback search
    for p in here.rglob("cfb_matched_output.csv"):
        return p
    return None


# -------------------------------------------------------------
# American odds utilities
# -------------------------------------------------------------
def american_to_prob(o):
    try:
        o = float(o)
    except:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return abs(o) / (abs(o) + 100.0)

def prob_to_decimal(p):
    if p is None or p <= 0:
        return None
    return 1.0 / p

def prob_to_american(p):
    if p <= 0 or p >= 1:
        return None
    if p < 0.5:
        return round(100 * (1 - p) / p)
    else:
        return round(-100 * p / (1 - p))


# -------------------------------------------------------------
# Vig-curve fallback (same as NFL/NBA boards)
# -------------------------------------------------------------
def vigcurve_fair_prob_from_single_side(odds):
    """Gentle fallback fair prob for single-side only cases."""
    try:
        o = float(odds)
    except:
        return None

    p = american_to_prob(o)
    if p is None:
        return None

    # base margin near even money ~4.5%, decays for long odds
    base = 0.045
    decay = min(1.0, 100 / (100 + abs(o)))
    margin = base * decay

    p_fair = p * (1 - margin)
    return max(min(p_fair, 0.999), 1e-6)


# -------------------------------------------------------------
# Pair-first fair model
# -------------------------------------------------------------
def compute_pair_first_fair(event_df, side_team, opp_team, market, line_val):
    """
    Given all rows for this event+market+line,
    find implied probabilities for side vs opp across books.
    """
    # Extract all side/opp rows
    side_rows = event_df[
        (event_df["_side"].str.lower() == str(side_team).lower())
    ]
    opp_rows = event_df[
        (event_df["_side"].str.lower() == str(opp_team).lower())
    ]

    if side_rows.empty or opp_rows.empty:
        return None

    # Collect all valid price pairs
    pair_ps = []
    for _, srow in side_rows.iterrows():
        for _, orow in opp_rows.iterrows():
            # Match same line for Totals & Spread
            if market != "Moneyline":
                try:
                    sl = float(srow["_line_num"])
                    ol = float(orow["_line_num"])
                    if abs(sl - ol) > 0.01:
                        continue
                except:
                    continue

            s_price = srow["Best Odds"] if pd.notna(srow["Best Odds"]) else None
            o_price = orow["Best Odds"] if pd.notna(orow["Best Odds"]) else None
            if s_price is None or o_price is None:
                continue
            try:
                sp, op = float(s_price), float(o_price)
            except:
                continue

            p_side = american_to_prob(sp)
            p_opp = american_to_prob(op)
            if p_side and p_opp and (p_side + p_opp) > 0:
                fair_p = p_side / (p_side + p_opp)
                pair_ps.append(fair_p)

    if not pair_ps:
        return None

    # Average the pair-first fair probabilities
    return float(np.mean(pair_ps))


# -------------------------------------------------------------
# Opponent team inference
# -------------------------------------------------------------
def infer_opponent(event, team):
    """Infer opponent team from 'A @ B' event."""
    try:
        left, right = [x.strip() for x in event.split("@")]
    except:
        return None
    if team.lower() == left.lower():
        return right
    if team.lower() == right.lower():
        return left
    return None


# -------------------------------------------------------------
# Core app
# -------------------------------------------------------------
def run():
    # Load CSV
    csv_path = _find_csv()
    if not csv_path:
        st.error("cfb_matched_output.csv not found.")
        return
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Timestamp display
    ts = datetime.fromtimestamp(csv_path.stat().st_mtime, pytz.utc)
    eastern = ts.astimezone(pytz.timezone("US/Eastern")).strftime("%Y-%m-%d %I:%M %p %Z")
    st.caption(f"Odds last updated: {eastern}")

    # ---------------------------------------------------------
    # Normalize column names your scraper produces
    # ---------------------------------------------------------
    df = df.rename(columns={
        "game": "Event",
        "bet_type": "Bet Type",
        "line": "Line",
        "selection": "Selection",
        "opponent": "Opponent",
        "best_book": "Best Book",
        "best_odds": "Best Odds",
        "value_ratio": "Line vs. Average"
    })

    # Clean market type
    def norm_market(bt):
        s = str(bt).lower()
        if "money" in s:
            return "Moneyline"
        if "over" in s or "under" in s or "total" in s:
            return "Total"
        return "Spread"

    df["Market"] = df["Bet Type"].apply(norm_market)

    # ---------------------------------------------------------
    # Establish sportsbook allowlist (EXCLUDE others)
    # ---------------------------------------------------------
    include_books = [
        "BetMGM",
        "BetRivers",
        "Bovada",
        "DraftKings",
        "FanDuel",
        "WilliamhillUs",   # will be renamed to Caesars
        "HardRock",
        "ESPNBet",
    ]

    # Extract sportsbook columns from CSV
    book_cols = []
    for c in df.columns:
        if c in include_books:
            book_cols.append(c)

    # Rename WilliamHill → Caesars in the UI *only*
    df = df.rename(columns={"WilliamhillUs": "Caesars"})
    book_cols = ["Caesars" if x == "WilliamhillUs" else x for x in book_cols]

    # ---------------------------------------------------------
    # Compute Best Book cleanly from allowed books only
    # ---------------------------------------------------------
    def best_book_row(r):
        best_name, best_dec, best_amer = None, None, None
        for b in book_cols:
            val = r.get(b, None)
            try:
                val_f = float(val)
            except:
                continue
            dec = prob_to_decimal(american_to_prob(val_f))
            if dec is None:
                continue
            if best_dec is None or dec > best_dec:
                best_dec = dec
                best_amer = val_f
                best_name = b
        return pd.Series({"BestBookCol": best_name, "Best Odds": best_amer})

    bb = df.apply(best_book_row, axis=1)
    df["BestBookCol"] = bb["BestBookCol"]
    df["Best Odds"] = bb["Best Odds"]

    # ---------------------------------------------------------
    # Side label
    # ---------------------------------------------------------
    df["_side"] = df["Selection"].astype(str)
    df["_line_num"] = pd.to_numeric(df["Line"], errors="coerce")

    # ---------------------------------------------------------
    # EV calculation (pair-first → vigcurve fallback)
    # ---------------------------------------------------------
    def calc_ev(r):
        best_odds = r["Best Odds"]
        if best_odds is None or not np.isfinite(best_odds):
            return np.nan

        event = r["Event"]
        market = r["Market"]
        side = r["_side"]
        opp = infer_opponent(event, side)

        event_df = df[df["Event"] == event]

        # Pair-first fair probability
        fair_p = compute_pair_first_fair(event_df, side, opp, market, r["_line_num"])
        if fair_p is None:
            # fallback to vig curve
            fair_p = vigcurve_fair_prob_from_single_side(best_odds)

        fair_dec = prob_to_decimal(fair_p)
        best_dec = prob_to_decimal(american_to_prob(best_odds))

        if fair_dec and best_dec:
            return (best_dec / fair_dec - 1) * 100
        return np.nan

    df["Implied EV (%)"] = df.apply(calc_ev, axis=1)

    # ---------------------------------------------------------
    # Sidebar Filters
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("Filters")

        events = ["All"] + sorted(df["Event"].unique())
        sel_event = st.selectbox("Game", events, 0)

        markets = ["All"] + sorted(df["Market"].unique())
        sel_market = st.selectbox("Market", markets, 0)

        books = ["All"] + book_cols
        sel_book = st.selectbox("Best Book", books, 0)

        min_books = st.number_input("Min. books posting this line", min_value=1,
                                    max_value=len(book_cols), value=2, step=1)

    # ---------------------------------------------------------
    # Apply filters
    # ---------------------------------------------------------
    if sel_event != "All":
        df = df[df["Event"] == sel_event]

    if sel_market != "All":
        df = df[df["Market"] == sel_market]

    if sel_book != "All":
        df = df[df["BestBookCol"] == sel_book]

    # Book count filter
    def count_valid_books(r):
        cnt = 0
        for b in book_cols:
            if pd.notna(r.get(b)) and str(r.get(b)).strip() != "":
                cnt += 1
        return cnt

    df["#Books"] = df.apply(count_valid_books, axis=1)
    df = df[df["#Books"] >= int(min_books)]

    # ---------------------------------------------------------
    # Final render
    # ---------------------------------------------------------
    df["Line vs. Average (%)"] = (pd.to_numeric(df["Line vs. Average"], errors="coerce") - 1) * 100

    df["Best Odds"] = df["Best Odds"].apply(lambda x: f"{int(x):+d}" if pd.notna(x) else "")

    # Format book odds
    for b in book_cols:
        df[b] = df[b].apply(lambda x: f"{int(float(x)):+d}" if pd.notna(x) else "")

    show_cols = ["Event", "Bet Type", "Line", "Selection"] + book_cols + [
        "Line vs. Average (%)", "Implied EV (%)"
    ]

    out = df[show_cols].copy()
    out = out.sort_values("Implied EV (%)", ascending=False)

    st.dataframe(out, use_container_width=True)


# -------------------------------------------------------------
# Run
# -------------------------------------------------------------
if __name__ == "__main__":
    run()
