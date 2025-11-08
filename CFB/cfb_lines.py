# app_cfb_game_lines.py
# College Football — Game Lines (Spreads • Moneylines • Totals)
# Drop-in replacement for your Streamlit board.
#
# Reads cfb_matched_output.csv (written by your odds job), then:
#   - excludes certain books from display/calcs
#   - adds/renames book columns (ESPNBet, Caesars, Hard Rock)
#   - recomputes Best Book ONLY from included books
#   - computes pair-first fair odds & Implied EV% with vig-curve fallback
#   - fixes Best Book filter bug for +money +EV lines

import os
import math
import pandas as pd
import numpy as np
import streamlit as st

# ---------- CONFIG ----------
DATA_FILE = os.path.join(os.path.dirname(__file__), "cfb_matched_output.csv")

# Column names in your CSV look like "FanDuel_Odds", "DraftKings_Odds", etc.
# Map API/book keys to a normalized, human-display name (and to expected CSV column).
CSV_BOOKS_MAP = {
    "BetMGM": "BetMGM_Odds",
    "BetRivers": "BetRivers_Odds",
    "Bovada": "Bovada_Odds",
    "DraftKings": "DraftKings_Odds",
    "FanDuel": "FanDuel_Odds",
    # Some boards output William Hill as "WilliamhillUs_Odds" – we rename to Caesars
    "WilliamhillUs": "WilliamhillUs_Odds",   # will display as "Caesars"
    # Optional / newer books (may or may not exist yet in your CSV):
    "ESPNBet": "ESPNBet_Odds",
    "HardRock": "HardRock_Odds",
    "Bet365": "Bet365_Odds",
    "WynnBET": "WynnBET_Odds",
    "PointsBet": "PointsBet_Odds",
    "Unibet": "Unibet_Odds",
}

# Books to EXCLUDE everywhere (display + calcs)
EXCLUDE_BOOKS_DISPLAY_AND_CALC = {"BetOnlineAG", "BetUS", "Lowvig", "MyBookieAG"}

# Friendly display renames (only surface names you include)
DISPLAY_RENAMES = {
    "WilliamhillUs": "Caesars",
    "WynnBET": "WynnBET",
    "PointsBet": "PointsBet",
    "ESPNBet": "ESPNBet",
    "HardRock": "Hard Rock",
}

# ---------- PRICING UTILS ----------
def american_to_decimal(odds: float | int | str | None) -> float | None:
    if odds in (None, "", np.nan):
        return None
    try:
        a = float(odds)
    except Exception:
        return None
    if a > 0:
        return 1.0 + (a / 100.0)
    if a < 0:
        return 1.0 + (100.0 / abs(a))
    return None

def american_to_prob(odds: float | int | str | None) -> float | None:
    if odds in (None, "", np.nan):
        return None
    try:
        a = float(odds)
    except Exception:
        return None
    if a > 0:
        return 100.0 / (a + 100.0)
    if a < 0:
        return abs(a) / (abs(a) + 100.0)
    return None

def prob_to_american(p: float) -> float | None:
    if p is None or p <= 0 or p >= 1:
        return None
    if p < 0.5:
        # positive American
        return round(100.0 * (1.0 - p) / p)
    else:
        # negative American
        return round(-100.0 * p / (1.0 - p))

def decimal_from_prob(p: float) -> float | None:
    if p is None or p <= 0:
        return None
    return 1.0 / p

def implied_ev_pct(offer_dec: float | None, fair_dec: float | None) -> float | None:
    if not offer_dec or not fair_dec:
        return None
    return (offer_dec / fair_dec - 1.0) * 100.0

# Mild “vig curve” fallback when a clean pair isn’t available.
# This gently reduces the implied probability by a margin that decays at long odds.
def fair_prob_from_single_side_vigcurve(odds: float) -> float | None:
    p = american_to_prob(odds)
    if p is None:
        return None
    # target margin ~4.5% around even money, decays for tails
    base = 0.045
    # decay factor: lower margin as price drifts away from even
    decay = min(1.0, 100.0 / (100.0 + abs(float(odds))))
    margin = base * decay
    # remove margin proportionally
    p_fair = p * (1.0 - margin)
    # renormalize softly, capped
    p_fair = max(min(p_fair, 0.999), 1e-6)
    return p_fair

# Pair-first fair odds from market-average paired prices (A vs B), de-vigged by normalization.
def pair_first_fair_prob(avg_side_prob: float | None, avg_opp_prob: float | None) -> float | None:
    if avg_side_prob is None or avg_opp_prob is None:
        return None
    total = avg_side_prob + avg_opp_prob
    if total <= 0:
        return None
    return avg_side_prob / total

# Given a row, figure out which selection is the “opposite side” key used to gather market averages.
def identify_opposite(row: pd.Series) -> str:
    bt = row["bet_type"]
    sel = str(row["selection"])
    opp = str(row["opponent"])
    if bt == "Total":
        # selection is "Over"/"Under"
        return "Under" if sel.lower() == "over" else "Over"
    elif bt == "Spread":
        # Spread: opponent team (same line)
        return opp
    elif bt == "Moneyline":
        # Moneyline: opponent team
        return opp
    return opp  # default

# ---------- DATA LOAD ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names we need
    # Some pipelines may have "line" as empty string for moneyline – keep as is
    # Ensure common columns exist
    expected = ["game", "kickoff_et", "bet_type", "selection", "opponent", "line"]
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    return df

# ---------- BOOK HANDLING ----------
def discover_books(df: pd.DataFrame) -> list[str]:
    # find all *_Odds columns
    return [c.replace("_Odds", "") for c in df.columns if c.endswith("_Odds")]

def included_book_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    books_in_file = discover_books(df)
    pairs: list[tuple[str, str]] = []
    for disp_name, csv_col in CSV_BOOKS_MAP.items():
        if disp_name in EXCLUDE_BOOKS_DISPLAY_AND_CALC:
            continue
        # Present?
        if csv_col in df.columns:
            pairs.append((disp_name, csv_col))
    return pairs

def rename_for_display(name: str) -> str:
    return DISPLAY_RENAMES.get(name, name)

# ---------- FAIR & EV CALC ----------
def compute_ev(df: pd.DataFrame, book_cols: list[tuple[str,str]]) -> pd.DataFrame:
    # Build helper frames of probabilities for every included book
    prob_cols = {}
    for disp, col in book_cols:
        pcol = f"__prob__{disp}"
        prob_cols[disp] = pcol
        df[pcol] = df[col].apply(american_to_prob)

    # Determine average implied probs for both sides within each (game, bet_type, line-point) group
    # Keys that define a *paired* market:
    group_keys = ["game", "bet_type", "line"]

    # Average probability for the current selection across books (included only)
    df["__avg_prob_this__"] = df[[prob_cols[d] for d, _ in book_cols]].mean(axis=1, skipna=True)

    # To get the opponent rows, self-join on (game, bet_type, line) with selection==opponent and vice versa
    side = df[group_keys + ["selection", "__avg_prob_this__"]].rename(
        columns={"selection": "__side__", "__avg_prob_this__": "__avg_prob_side__"}
    )
    opp = df[group_keys + ["selection", "__avg_prob_this__"]].rename(
        columns={"selection": "__opp__", "__avg_prob_this__": "__avg_prob_opp__"}
    )

    merged = df.merge(
        # map each row’s “opponent” name to the other row’s avg prob (same group)
        opp,
        left_on=group_keys + ["opponent"],
        right_on=group_keys + ["__opp__"],
        how="left",
        suffixes=("", "_y"),
    )
    merged = merged.drop(columns=["__opp__"], errors="ignore")
    merged = merged.merge(
        side,
        left_on=group_keys + ["selection"],
        right_on=group_keys + ["__side__"],
        how="left",
        suffixes=("", "_z"),
    )
    merged = merged.drop(columns=["__side__"], errors="ignore")

    # Pair-first: fair probability = avg_prob_side / (avg_prob_side + avg_prob_opp)
    merged["__pair_fair_prob__"] = merged.apply(
        lambda r: pair_first_fair_prob(r["__avg_prob_side__"], r["__avg_prob_opp__"]),
        axis=1,
    )

    # Fallback: vig curve on this offer’s own price if needed
    # We'll generate a fair_prob_fallback for each row from the Best Book price later,
    # but we also keep a per-row fallback based on ANY visible price in this row.
    any_offer_prob = df[[prob_cols[d] for d, _ in book_cols]].bfill(axis=1).iloc[:, 0]
    any_offer_odds = df[[c for _, c in book_cols]].bfill(axis=1).iloc[:, 0]
    merged["__single_side_fair_prob__"] = any_offer_odds.apply(
        lambda o: fair_prob_from_single_side_vigcurve(o) if pd.notna(o) and o != "" else None
    )

    # Choose fair prob: pair-first, else single-side vig curve
    merged["__fair_prob__"] = merged["__pair_fair_prob__"].combine_first(merged["__single_side_fair_prob__"])

    # For Implied EV we need the *offer* (best among included books), so compute offer now
    def best_from_included(row):
        best_name, best_col, best_dec, best_amer = None, None, None, None
        for disp, col in book_cols:
            val = row.get(col, "")
            dec = american_to_decimal(val)
            if dec is None:
                continue
            if best_dec is None or dec > best_dec:
                best_name, best_col, best_dec, best_amer = disp, col, dec, val
        return best_name, best_col, best_dec, best_amer

    best = df.apply(best_from_included, axis=1, result_type="expand")
    best.columns = ["best_book_name", "best_book_col", "best_decimal", "best_american"]
    merged = pd.concat([merged, best], axis=1)

    # If pair-first failed and we need a fair prob just for EV, do a fallback on the specific best price
    def row_fair_prob(r):
        p = r["__fair_prob__"]
        if p and 0 < p < 1:
            return p
        amer = r["best_american"]
        if amer in ("", None) or (isinstance(amer, float) and np.isnan(amer)):
            return None
        return fair_prob_from_single_side_vigcurve(amer)

    merged["__fair_prob_final__"] = merged.apply(row_fair_prob, axis=1)
    merged["fair_decimal"] = merged["__fair_prob_final__"].apply(decimal_from_prob)

    # Implied EV %
    merged["implied_ev_pct"] = merged.apply(
        lambda r: implied_ev_pct(r["best_decimal"], r["fair_decimal"]),
        axis=1,
    )

    return merged

# ---------- UI ----------
st.set_page_config(page_title="College Football — Game Lines", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:0.25rem;'>College Football — Game Lines</h1>"
    "<div>Game Spreads · Moneylines · Totals (pair-first fair pricing with vig-curve fallback)</div>",
    unsafe_allow_html=True,
)

df = load_data(DATA_FILE)

# Build included books from what actually exists in the file
book_pairs = included_book_columns(df)
# Remove any excluded that slipped into CSV_BOOKS_MAP (safety)
book_pairs = [(d, c) for (d, c) in book_pairs if d not in EXCLUDE_BOOKS_DISPLAY_AND_CALC]

# Rename/alias display
book_display_cols = []
for disp, col in book_pairs:
    nice = rename_for_display(disp)
    book_display_cols.append((nice, col))

# Recompute Best Book **only** from included books to fix filtering bugs
def recompute_best_book_only_included(row):
    best_name, best_col, best_dec = None, None, None
    for nice, col in book_display_cols:
        dec = american_to_decimal(row.get(col))
        if dec is None:
            continue
        if best_dec is None or dec > best_dec:
            best_name, best_col, best_dec = nice, col, dec
    return pd.Series({"best_book": best_name or "", "best_odds": row.get(best_col) if best_col else ""})

best_fix = df.apply(recompute_best_book_only_included, axis=1)
df = pd.concat([df.drop(columns=[c for c in ["best_book", "best_odds"] if c in df.columns]), best_fix], axis=1)

# Compute EV (pair-first + fallback)
priced = compute_ev(df.copy(), book_pairs)

# Build filter widgets
with st.sidebar:
    st.header("Filters")
    # Game filter
    games = ["All"] + sorted(priced["game"].dropna().unique().tolist())
    sel_game = st.selectbox("Game", games, index=0)

    markets = ["All", "Spread", "Moneyline", "Total"]
    sel_market = st.selectbox("Market", markets, index=0)

    bestbook_opts = ["All"] + sorted({rename_for_display(x) for (x, _) in book_pairs})
    sel_best_book = st.selectbox("Best Book", bestbook_opts, index=0)

    min_books = st.number_input("Min. books posting this line", min_value=1, max_value=20, value=2, step=1)

    compact = st.toggle("Compact\nmobile\nmode", value=False)

# Apply filters
working = priced.copy()

# Min books posting (count across included columns)
included_cols = [col for _, col in book_pairs]
working["__book_count__"] = working[included_cols].apply(lambda r: (~r.isna() & (r.astype(str) != "")).sum(), axis=1)
working = working[working["__book_count__"] >= int(min_books)]

if sel_game != "All":
    working = working[working["game"] == sel_game]

if sel_market != "All":
    working = working[working["bet_type"] == sel_market]

if sel_best_book != "All":
    working = working[working["best_book"] == sel_best_book]

# Display table
# Assemble final columns
prefix = ["game", "kickoff_et", "bet_type", "selection", "opponent", "line"]
book_cols_out = [c for (nice, c) in book_display_cols]
suffix = [
    "best_book",
    "best_american",
    "best_decimal",
    "fair_decimal",
    "implied_ev_pct",
]

# Pretty rename columns for display
nice_names = {
    "kickoff_et": "Kickoff (ET)",
    "bet_type": "Bet Type",
    "selection": "Selection",
    "opponent": "Opponent",
    "line": "Line",
    "best_american": "Best Odds",
    "best_decimal": "Best Dec",
    "fair_decimal": "Fair Dec",
    "implied_ev_pct": "Implied EV (%)",
    "__book_count__": "# Books",
}
for nice, col in book_display_cols:
    nice_names[col] = nice

# Build and sort
out = working.copy()

# Ensure moneylines show blank line cell
out.loc[out["bet_type"] == "Moneyline", "line"] = out.loc[out["bet_type"] == "Moneyline", "line"].replace({0.0: ""})

# Keep only final columns
final_cols = prefix + book_cols_out + suffix
for c in suffix:
    if c not in out.columns:
        out[c] = ""

out = out[final_cols]

# Sort by EV descending within moneylines first (then spreads/totals)
sort_keys = [
    (out["bet_type"] == "Moneyline").astype(int),  # Moneyline first
    out["implied_ev_pct"].fillna(-1e9),
]
out = out.sort_values(by=[k for k in range(len(sort_keys))], key=lambda s: sort_keys.pop(0), ascending=False)

# Formatting
fmt = {
    "best_decimal": "{:.3f}",
    "fair_decimal": "{:.3f}",
    "implied_ev_pct": lambda x: "" if pd.isna(x) else f"{x:.1f}%",
}

styled = out.rename(columns=nice_names).style

# Highlight best prices per row among included books
def highlight_best_row(row):
    vals = []
    best_val = None
    for _, col in book_display_cols:
        v = row.get(nice_names.get(col, col), "")
        if v == "" or pd.isna(v):
            vals.append(None)
        else:
            dec = american_to_decimal(v)
            vals.append(dec)
            if dec is not None:
                best_val = max(best_val or dec, dec)
    colors = []
    for dec in vals:
        if dec is not None and best_val is not None and abs(dec - best_val) < 1e-9:
            colors.append("background-color: #c8f7c5")  # green highlight
        else:
            colors.append("")
    return colors

styled = styled.apply(highlight_best_row, axis=1, subset=[nice_names[c] for _, c in book_display_cols])

# Apply numeric formats
for col, f in fmt.items():
    disp = nice_names.get(col, col)
    styled = styled.format({disp: f})

# Compact mode tweaks (narrow the first two columns)
if compact:
    styled = styled.set_table_styles([
        {"selector": "th.col0, td.col0", "props": "min-width: 140px; max-width: 180px;"},
        {"selector": "th.col1, td.col1", "props": "min-width: 120px; max-width: 140px;"},
    ], overwrite=False)

st.caption("Odds last updated: the time your CSV job last ran")
st.dataframe(styled, use_container_width=True)
