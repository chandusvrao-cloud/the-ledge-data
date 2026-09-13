"""Base/pivot finders for the pattern-screens suite: Multi-year breakouts, IPO
base, Blue sky, and VCP.

Definition shared by all of them (matches BananaPatterns' own #method
description): the pivot is the HIGHEST High that occurred >= min_base_weeks ago
and has not been closed above until (at earliest) some later breakout. Picking
the highest such high — not just the oldest — matters: a more recent, larger
high represents the real overhead resistance and must take priority over an
older, smaller one.

Debugging note (keep this in mind before changing the validity condition):
an earlier version required the pivot to *never* be closed above (only a small
grace tail excepted). That made it structurally impossible to ever detect a
stock that broke out and kept climbing for more than a few sessions — the old
pivot would look "broken" and get thrown out, so climbing/played_out came back
empty across the entire universe. The fix (still in place below, in
_first_breakout/_stage): validity only requires the base be unbroken *up to*
the first breakout; whatever happens after breakout doesn't disqualify the
candidate — it's what determines the stage.
"""
import numpy as np
import pandas as pd

FRESH_BREAKOUT_SESSIONS = 5   # broke out within this many sessions ago -> "fresh_breakout" stage


def _first_breakout(closes, pivot, pivot_idx):
    """First index j > pivot_idx where closes[j] > pivot, or None if price has
    stayed at/under the pivot the whole time since (still forming)."""
    n = len(closes)
    if pivot_idx + 1 >= n:
        return None
    post = closes[pivot_idx + 1:]
    above = post > pivot
    if not above.any():
        return None
    return pivot_idx + 1 + int(np.argmax(above))


def _stage(pivot, closes, breakout_idx, last_close):
    """Stage reflects TODAY's position, not just whether a breakout happened at
    some point. A stock that broke out and has since fallen back under the
    pivot is "played_out" (failed/faded breakout), never "fresh_breakout" or
    "forming" — those both require last_close relative to pivot to actually
    match ("forming" = never broke out at all; "fresh_breakout"/"climbing" both
    require last_close > pivot right now)."""
    n = len(closes)
    pct_from_pivot = (last_close - pivot) / pivot * 100

    if breakout_idx is None:
        return "forming", pct_from_pivot, None

    sessions_since_breakout = (n - 1) - breakout_idx

    if last_close <= pivot:
        return "played_out", pct_from_pivot, sessions_since_breakout  # broke out, has since fallen back under

    if sessions_since_breakout <= FRESH_BREAKOUT_SESSIONS:
        return "fresh_breakout", pct_from_pivot, sessions_since_breakout

    sma20 = float(np.mean(closes[max(0, n - 20):]))
    stage = "climbing" if last_close > sma20 else "played_out"
    return stage, pct_from_pivot, sessions_since_breakout


def _find_pivot_idx(dates, highs, lows, closes, last_date, min_base_weeks, max_base_depth_pct):
    """Shared search: among indices old enough and with a reasonable base depth,
    return the one with the highest `highs[i]`. Returns None if none qualify."""
    n = len(closes)
    weeks_since = np.array([(last_date - d).days / 7.0 for d in dates])
    best_i, best_pivot = None, -1.0
    for i in range(n - 1):
        if weeks_since[i] < min_base_weeks:
            continue
        pivot_i = highs[i]
        base_low_i = float(lows[i:].min())
        depth_i = (pivot_i - base_low_i) / pivot_i * 100
        if depth_i > max_base_depth_pct:
            continue
        if pivot_i > best_pivot:
            best_pivot = pivot_i
            best_i = i
    return best_i, weeks_since


def _build_record(dates, highs, lows, closes, pivot_idx, weeks_since, near_pivot_pct, extra=None):
    pivot = float(highs[pivot_idx])
    breakout_idx = _first_breakout(closes, pivot, pivot_idx)
    base_low = float(lows[pivot_idx:].min())
    base_depth_pct = (pivot - base_low) / pivot * 100
    last_close = float(closes[-1])
    stage, pct_from_pivot, sessions_since_breakout = _stage(pivot, closes, breakout_idx, last_close)

    if abs(pct_from_pivot) > near_pivot_pct and stage == "forming":
        return None  # too far below pivot to be "near the trigger" — not surfaced

    rec = {
        "pivot": round(pivot, 2),
        "pivot_date": dates[pivot_idx].strftime("%Y-%m-%d"),
        "base_weeks": round(float(weeks_since[pivot_idx]), 1),
        "base_depth_pct": round(float(base_depth_pct), 1),
        "close": round(last_close, 2),
        "pct_from_pivot": round(float(pct_from_pivot), 1),
        "stage": stage,
        "sessions_since_breakout": sessions_since_breakout,
        "chart_from_idx": pivot_idx,
    }
    if extra:
        rec.update(extra)
    return rec


def find_long_base(dates, highs, lows, closes, last_date, min_base_weeks=52,
                    max_base_depth_pct=65.0, near_pivot_pct=20.0):
    """Powers Multi-year breakouts (default thresholds) and IPO base (relaxed
    min_base_weeks, restricted universe — see find_ipo_base)."""
    n = len(closes)
    if n < 30:
        return None
    pivot_idx, weeks_since = _find_pivot_idx(dates, highs, lows, closes, last_date,
                                              min_base_weeks, max_base_depth_pct)
    if pivot_idx is None:
        return None
    return _build_record(dates, highs, lows, closes, pivot_idx, weeks_since, near_pivot_pct)


def find_ipo_base(dates, highs, lows, closes, last_date, min_base_weeks=10):
    """Same mechanics as find_long_base, relaxed minimum base duration (a young
    stock can't have a 52-week base yet) and a looser depth allowance (first
    post-IPO corrections often run deep). Caller restricts the universe to
    recently listed symbols (see build_screens.py)."""
    return find_long_base(dates, highs, lows, closes, last_date,
                           min_base_weeks=min_base_weeks, max_base_depth_pct=70.0, near_pivot_pct=25.0)


def find_blue_sky(dates, highs, lows, closes, last_date, min_base_weeks=4,
                   max_base_depth_pct=25.0, near_pivot_pct=6.0):
    """Pivot = the all-time high in available history (no more-significant peak
    exists to search for, unlike find_long_base — there's only one candidate).
    Qualifies only if the pullback since that high has been shallow (little to
    no overhead supply left)."""
    n = len(closes)
    if n < 30:
        return None
    pivot_idx = int(np.argmax(highs))
    weeks_since = np.array([(last_date - d).days / 7.0 for d in dates])
    if weeks_since[pivot_idx] < min_base_weeks:
        return None
    pivot = float(highs[pivot_idx])
    base_low = float(lows[pivot_idx:].min())
    depth = (pivot - base_low) / pivot * 100
    if depth > max_base_depth_pct:
        return None
    return _build_record(dates, highs, lows, closes, pivot_idx, weeks_since, near_pivot_pct)


def find_52w_high(dates, highs, lows, closes, last_date, window_days=252,
                   max_base_depth_pct=90.0, near_pivot_pct=8.0):
    """Classic 52-week-high screener: pivot = the highest High within the
    trailing ~252 sessions only (not the full available history — that's
    find_blue_sky's job). No minimum age and a loose depth cap by design: this
    is meant to be the broad, standard "new highs list," not a quality-filtered
    breakout screen. Reuses find_long_base's exact shape (pivot/breakout/stage)
    with the search window truncated to the trailing year instead of expanded
    to find the highest *unbroken* high — here we just want the trailing max."""
    n = len(closes)
    if n < 30:
        return None
    start = max(0, n - window_days)
    local_idx = int(np.argmax(highs[start:]))
    pivot_idx = start + local_idx
    weeks_since = np.array([(last_date - d).days / 7.0 for d in dates])
    pivot = float(highs[pivot_idx])
    base_low = float(lows[pivot_idx:].min())
    depth = (pivot - base_low) / pivot * 100
    if depth > max_base_depth_pct:
        return None
    return _build_record(dates, highs, lows, closes, pivot_idx, weeks_since, near_pivot_pct)


def _local_minima(closes, window=3):
    """Indices i where closes[i] is the minimum within +/- window — simple
    swing-low detector used by find_vcp's contraction check."""
    n = len(closes)
    idxs = []
    for i in range(window, n - window):
        seg = closes[i - window:i + window + 1]
        if closes[i] == seg.min() and np.argmin(seg) == window:
            idxs.append(i)
    return idxs


def find_vcp(dates, highs, lows, closes, last_date, min_base_weeks=5,
              max_base_depth_pct=50.0, near_pivot_pct=15.0, min_contractions=2):
    """Same base/pivot search as find_long_base (shorter minimum duration, less
    depth tolerance — VCP bases are typically tighter and shorter than the
    Multi-year screen's), plus a volatility-contraction check: within the base,
    successive pullbacks (swing-high -> swing-low) must get shallower at least
    `min_contractions` times in a row (the classic VCP "T1 > T2 > T3" signature)."""
    n = len(closes)
    if n < 30:
        return None
    pivot_idx, weeks_since = _find_pivot_idx(dates, highs, lows, closes, last_date,
                                              min_base_weeks, max_base_depth_pct)
    if pivot_idx is None:
        return None

    breakout_idx = _first_breakout(closes, float(highs[pivot_idx]), pivot_idx)
    seg_end = breakout_idx if breakout_idx is not None else n
    segment = closes[pivot_idx:seg_end]
    if len(segment) < 15:
        return None

    lows_idx = _local_minima(segment, window=3)
    depths = []
    for li in lows_idx:
        preceding_high = float(segment[:li + 1].max())
        trough = float(segment[li])
        if preceding_high <= 0:
            continue
        depths.append((preceding_high - trough) / preceding_high * 100)

    contractions = 0
    for a, b in zip(depths, depths[1:]):
        if b < a * 0.85:
            contractions += 1
    if contractions < min_contractions:
        return None

    return _build_record(dates, highs, lows, closes, pivot_idx, weeks_since, near_pivot_pct,
                          extra={"contraction_depths": [round(d, 1) for d in depths]})


def find_dma_breakout(dates, closes, last_date, window, near_pct=5.0, lookback_days=90):
    """Pivot here is a MOVING level (the SMA itself), unlike the fixed-price
    pivots above — "breakout" means the most recent close-crosses-above-SMA
    event. `chart_from_idx` + a `sma_window` marker let the frontend draw the
    actual SMA curve instead of a flat dashed line for this screen."""
    n = len(closes)
    if n < window + 5:
        return None
    sma = pd.Series(closes).rolling(window).mean().to_numpy()
    valid_start = window - 1
    search_start = max(valid_start + 1, n - lookback_days)

    breakout_idx = None
    for i in range(search_start, n):
        if np.isnan(sma[i]) or np.isnan(sma[i - 1]):
            continue
        if closes[i - 1] <= sma[i - 1] and closes[i] > sma[i]:
            breakout_idx = i  # keep the LAST (most recent) crossing

    last_close = float(closes[-1])
    last_sma = float(sma[-1])
    pct_from_sma = (last_close - last_sma) / last_sma * 100
    chart_start = max(0, n - 90)

    if breakout_idx is None:
        if last_close <= last_sma and abs(pct_from_sma) <= near_pct:
            return {
                "pivot": round(last_sma, 2), "pivot_date": dates[-1].strftime("%Y-%m-%d"),
                "base_weeks": None, "base_depth_pct": None, "close": round(last_close, 2),
                "pct_from_pivot": round(pct_from_sma, 1), "stage": "forming",
                "sessions_since_breakout": None, "chart_from_idx": chart_start, "sma_window": window,
            }
        return None

    sessions_since_breakout = (n - 1) - breakout_idx
    if last_close <= last_sma:
        stage = "played_out"
    elif sessions_since_breakout <= FRESH_BREAKOUT_SESSIONS:
        stage = "fresh_breakout"
    else:
        stage = "climbing"

    return {
        "pivot": round(last_sma, 2), "pivot_date": dates[breakout_idx].strftime("%Y-%m-%d"),
        "base_weeks": None, "base_depth_pct": None, "close": round(last_close, 2),
        "pct_from_pivot": round(pct_from_sma, 1), "stage": stage,
        "sessions_since_breakout": sessions_since_breakout,
        "chart_from_idx": max(chart_start, breakout_idx - 20), "sma_window": window,
    }


def find_pole_flag(dates, closes, vols, last_date,
                    pole_min_days=3, pole_max_days=10, pole_min_pct=15.0, pole_vol_mult=1.3,
                    flag_min_days=5, flag_max_days=15, flag_max_range_pct=12.0,
                    flag_max_retrace_pct=50.0, flag_vol_max_mult=0.85,
                    lookback_days=45, near_pivot_pct=12.0, sma_window=50):
    """Textbook pole-and-flag: a sharp pole (>=pole_min_pct in pole_min_days..
    pole_max_days, on volume) followed by a tight, low-volume flag consolidation
    that hasn't yet broken meaningfully above the pole high. Reuses the same
    pivot=pole_high / _first_breakout / _stage machinery as the base-pivot
    screens once the flag is identified — same short-term heuristic definition
    as the (unbacktested) standalone pole_flag_scanner.py this account already
    runs, ported into the shared engine instead of re-derived from scratch."""
    n = len(closes)
    min_needed = sma_window + lookback_days + pole_max_days + flag_max_days
    if n < min_needed:
        return None

    sma = pd.Series(closes).rolling(sma_window).mean().to_numpy()
    if np.isnan(sma[-1]) or np.isnan(sma[-21]):
        return None
    if not (closes[-1] > sma[-1] and sma[-1] > sma[-21]):
        return None  # uptrend gate

    search_start = max(sma_window, n - lookback_days - pole_max_days - flag_max_days)
    best = None  # prefer the most recently completed flag

    for pole_start in range(search_start, n - flag_min_days - pole_min_days):
        pole_start_close = closes[pole_start]
        if pole_start_close <= 0:
            continue
        baseline_start = max(0, pole_start - 20)
        if pole_start <= baseline_start:
            continue
        baseline_vol = float(vols[baseline_start:pole_start].mean())
        if baseline_vol <= 0:
            continue

        for pole_len in range(pole_min_days, pole_max_days + 1):
            pole_end = pole_start + pole_len - 1
            if pole_end >= n - flag_min_days:
                break
            pole_end_close = closes[pole_end]
            pole_gain = (pole_end_close - pole_start_close) / pole_start_close * 100
            if pole_gain < pole_min_pct:
                continue
            pole_avg_vol = float(vols[pole_start:pole_end + 1].mean())
            if pole_avg_vol < pole_vol_mult * baseline_vol:
                continue
            pole_high = float(closes[pole_start:pole_end + 1].max())
            pole_move = pole_high - pole_start_close
            if pole_move <= 0:
                continue

            for flag_len in range(flag_min_days, flag_max_days + 1):
                flag_start = pole_end + 1
                flag_end = min(flag_start + flag_len - 1, n - 1)
                if flag_end - flag_start + 1 < flag_min_days:
                    continue
                flag_slice = closes[flag_start:flag_end + 1]
                flag_high = float(flag_slice.max())
                flag_low = float(flag_slice.min())
                if flag_high > pole_high * 1.02:
                    continue  # already broke out meaningfully during the flag itself
                flag_range_pct = (flag_high - flag_low) / pole_high * 100
                if flag_range_pct > flag_max_range_pct:
                    continue
                retrace_pct = (pole_high - flag_low) / pole_move * 100
                if retrace_pct > flag_max_retrace_pct:
                    continue
                flag_avg_vol = float(vols[flag_start:flag_end + 1].mean())
                if flag_avg_vol > flag_vol_max_mult * pole_avg_vol:
                    continue
                if best is None or flag_end > best[0]:
                    best = (flag_end, pole_high, pole_start, pole_gain, flag_len, flag_start)

    if best is None:
        return None

    flag_end, pole_high, pole_start, pole_gain, flag_len, flag_start = best
    breakout_idx = _first_breakout(closes, pole_high, flag_end)
    last_close = float(closes[-1])
    stage, pct_from_pivot, sessions_since_breakout = _stage(pole_high, closes, breakout_idx, last_close)
    if abs(pct_from_pivot) > near_pivot_pct and stage == "forming":
        return None

    weeks_since = (last_date - dates[pole_start]).days / 7.0
    flag_low = float(closes[flag_start:flag_end + 1].min())
    return {
        "pivot": round(pole_high, 2),
        "pivot_date": dates[pole_start].strftime("%Y-%m-%d"),
        "base_weeks": round(weeks_since, 1),
        "base_depth_pct": round((pole_high - flag_low) / pole_high * 100, 1),
        "close": round(last_close, 2),
        "pct_from_pivot": round(pct_from_pivot, 1),
        "stage": stage,
        "sessions_since_breakout": sessions_since_breakout,
        "pole_gain_pct": round(pole_gain, 1),
        "flag_days": flag_len,
        "chart_from_idx": pole_start,
    }
