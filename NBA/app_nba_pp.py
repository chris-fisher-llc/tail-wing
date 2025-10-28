def _z_score(row, delta=1e-6):
    """Robust z-score of best book implied prob vs. market median."""
    try:
        # get best book implied prob
        d_best = float(row.get("best_decimal", float("nan")))
        if pd.isna(d_best) or d_best <= 1:
            return float("nan")
        p_best = 1.0 / d_best

        # collect all implied probabilities across books
        probs = []
        for c in book_cols:
            v = row.get(c)
            if pd.isna(v):
                continue
            # handle both American (+120) and decimal (1.85) formats
            s = str(v).strip()
            if s == "":
                continue
            if s.startswith("+") or s.startswith("-"):
                try:
                    a = float(s)
                    if a > 0:
                        dec = 1 + (a / 100)
                    else:
                        dec = 1 + (100 / abs(a))
                    probs.append(1 / dec)
                except Exception:
                    continue
            else:
                try:
                    dec = float(s)
                    if dec > 1:
                        probs.append(1 / dec)
                except Exception:
                    continue

        if len(probs) < 3:
            return float("nan")

        series = pd.Series(probs)
        m = series.median()
        mad = (series - m).abs().median() * 1.4826  # robust MAD scale factor
        return (m - p_best) / (mad + delta)
    except Exception:
        return float("nan")
