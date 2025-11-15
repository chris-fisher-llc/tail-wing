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
    return max(min(p_fair, 0.999), 1e-_
