"""Orchestrator — runs all pattern screens over the real NSE universe and
writes screens_data.json (+ a dated snapshot for next-run diffing).

Screens: multiyear, vcp, bluesky, ipobase, poleflag, dma20/dma50/dma200.

Run: python build_screens.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pivot_engine
import record_screen
import screens_common as sc

HERE = Path(__file__).parent
SNAPSHOT_DIR = HERE / "snapshots"

MIN_RS_MULTIYEAR = 60
MIN_RS_VCP = 60
MIN_RS_BLUESKY = 70
IPO_MAX_AGE_DAYS = 730          # only symbols listed within this many days ago...
IPO_CACHE_START_BUFFER = pd.Timestamp("2023-10-01")  # ...and after this date, to exclude cache-start artifacts
CHART_MAX_POINTS = 260          # cap embedded daily chart history per stock (charts ship in separate per-screen files, not the main page)


def _symbol_frame(big, sym):
    return big[big["SYMBOL"] == sym].reset_index(drop=True)


def _finish_hit(rec, sym, row, g, price_pivot="__use_rec_pivot__"):
    """Common post-processing: attach sym/rs/turnover, build the click-to-chart
    payload (daily + weekly OHLC, from chart_from_idx forward), drop internal
    fields. `g` is this symbol's full daily frame (DATE, OPEN/HIGH/LOW/CLOSE_PRICE),
    reset to a plain 0..n-1 index so positions line up with the numpy arrays the
    detector functions worked from.

    `price_pivot` is the horizontal line drawn on the PRICE chart — defaults to
    rec["pivot"], which is correct everywhere rec["pivot"] is itself a price
    (all the base/pivot/DMA screens). The record-profit screen's "pivot" is a
    ₹Cr fundamentals figure, not a price — pass price_pivot=None there so the
    chart doesn't draw a nonsense horizontal line at a crore-scale Y value."""
    pivot_idx = rec.pop("chart_from_idx")
    sma_window = rec.pop("sma_window", None)
    n = len(g)

    breakout_date = None
    breakout_metrics = None
    if rec["sessions_since_breakout"] is not None:
        bi = (n - 1) - rec["sessions_since_breakout"]
        breakout_date = g["DATE"].iloc[bi]
        b_open, b_high, b_low, b_close = (float(g["OPEN_PRICE"].iloc[bi]), float(g["HIGH_PRICE"].iloc[bi]),
                                           float(g["LOW_PRICE"].iloc[bi]), float(g["CLOSE_PRICE"].iloc[bi]))
        prev_close = float(g["CLOSE_PRICE"].iloc[bi - 1]) if bi > 0 else None
        vol_base_start = max(0, bi - 20)
        baseline_vol = float(g["TTL_TRD_QNTY"].iloc[vol_base_start:bi].mean()) if bi > vol_base_start else None
        b_vol = float(g["TTL_TRD_QNTY"].iloc[bi])
        breakout_metrics = {
            "breakout_date": breakout_date.strftime("%Y-%m-%d"),
            "breakout_day_gain_pct": round((b_close - prev_close) / prev_close * 100, 1) if prev_close else None,
            "breakout_close_in_range": round((b_close - b_low) / (b_high - b_low), 2) if b_high > b_low else 0.5,
            "breakout_vol_ratio": round(b_vol / baseline_vol, 1) if baseline_vol else None,
        }

    daily_slice = g.iloc[pivot_idx:]
    if len(daily_slice) > CHART_MAX_POINTS:
        daily_slice = daily_slice.iloc[len(daily_slice) - CHART_MAX_POINTS:]

    def _breakout_offset(dates_series):
        if breakout_date is None:
            return None
        pos = int(np.searchsorted(dates_series.to_numpy(), np.datetime64(breakout_date)))
        return pos if 0 <= pos < len(dates_series) else None

    daily = {
        "o": [round(float(v), 2) for v in daily_slice["OPEN_PRICE"]],
        "h": [round(float(v), 2) for v in daily_slice["HIGH_PRICE"]],
        "l": [round(float(v), 2) for v in daily_slice["LOW_PRICE"]],
        "c": [round(float(v), 2) for v in daily_slice["CLOSE_PRICE"]],
        "v": [int(v) for v in daily_slice["TTL_TRD_QNTY"]],
    }
    breakout_offset_daily = _breakout_offset(daily_slice["DATE"])

    def _ma_slice(window):
        full = pd.Series(g["CLOSE_PRICE"].to_numpy()).rolling(window).mean()
        sl = full.iloc[daily_slice.index]
        return [None if pd.isna(v) else round(float(v), 2) for v in sl]

    # Always show 50/200-day MAs (matches how real Stage-2 chart tools like
    # BananaPatterns always overlay both) — computed from the FULL history so
    # early bars in the chart window still have a correct MA, not just from
    # where the chart happens to start. The DMA screens additionally get their
    # own specific window (e.g. 20-day for dma20) since 50/200 alone wouldn't
    # show what that screen is actually about.
    ma50 = _ma_slice(50) if n >= 50 else None
    ma200 = _ma_slice(200) if n >= 200 else None
    ma_extra = None
    if sma_window and sma_window not in (50, 200):
        ma_extra = {"window": sma_window, "line": _ma_slice(sma_window)}

    last_close = float(g["CLOSE_PRICE"].iloc[-1])
    prev_day_close = float(g["CLOSE_PRICE"].iloc[-2]) if n > 1 else None
    one_day_gain_pct = round((last_close - prev_day_close) / prev_day_close * 100, 1) if prev_day_close else None
    price_vs_50ma_pct = round((last_close - ma50[-1]) / ma50[-1] * 100, 1) if (ma50 and ma50[-1]) else None

    weekly_df = (g.set_index("DATE")
                  .resample("W-FRI")
                  .agg({"OPEN_PRICE": "first", "HIGH_PRICE": "max", "LOW_PRICE": "min",
                        "CLOSE_PRICE": "last", "TTL_TRD_QNTY": "sum"})
                  .dropna())
    weekly_from = g["DATE"].iloc[pivot_idx]
    weekly_df = weekly_df[weekly_df.index >= weekly_from]
    weekly = {
        "o": [round(float(v), 2) for v in weekly_df["OPEN_PRICE"]],
        "h": [round(float(v), 2) for v in weekly_df["HIGH_PRICE"]],
        "l": [round(float(v), 2) for v in weekly_df["LOW_PRICE"]],
        "c": [round(float(v), 2) for v in weekly_df["CLOSE_PRICE"]],
        "v": [int(v) for v in weekly_df["TTL_TRD_QNTY"]],
    }
    breakout_offset_weekly = _breakout_offset(pd.Series(weekly_df.index))

    chart_pivot = rec["pivot"] if price_pivot == "__use_rec_pivot__" else price_pivot

    rec["sym"] = sym
    rec["rs"] = None if pd.isna(row["rs"]) else int(row["rs"])
    rec["avg_turnover_cr"] = round(float(row["avg_turnover_cr"]), 1)
    rec["one_day_gain_pct"] = one_day_gain_pct
    rec["price_vs_50ma_pct"] = price_vs_50ma_pct
    if breakout_metrics:
        rec.update(breakout_metrics)
    rec["chart"] = {
        "daily": daily, "weekly": weekly, "pivot": chart_pivot,
        "breakout_offset_daily": breakout_offset_daily,
        "breakout_offset_weekly": breakout_offset_weekly,
        "ma50": ma50, "ma200": ma200, "ma_extra": ma_extra,
        "base_weeks": rec["base_weeks"],
    }
    return rec


def bucket_by_stage(hits):
    buckets = {"forming": [], "fresh_breakout": [], "climbing": [], "played_out": []}
    for h in hits:
        buckets[h["stage"]].append(h)
    for stage in buckets:
        buckets[stage].sort(key=lambda r: r["pct_from_pivot"])
    return buckets


def bucket_by_record_type(hits):
    buckets = {"both": [], "profit_only": [], "revenue_only": []}
    for h in hits:
        buckets[h["stage"]].append(h)
    for stage in buckets:
        buckets[stage].sort(key=lambda r: r["pct_from_pivot"], reverse=True)
    return buckets


def split_charts(buckets):
    """Pulls the (large) 'chart' payload out of each hit into a separate
    {sym: chart} map, leaving the summary record light. Charts ship as their
    own per-screen JSON file (fetched client-side on symbol click) instead of
    inflating the main page — with ~1100+ hits on the DMA screens, embedding
    every chart directly in the page pushed output.html past the 16MB artifact
    limit (34MB on the first attempt)."""
    charts = {}
    for rows in buckets.values():
        for r in rows:
            charts[r["sym"]] = r.pop("chart")
    return charts


def run_multiyear(big, uni_df, last_date):
    hits = []
    for sym, row in uni_df.iterrows():
        if not (row["above_200dma"] and row["dma_turning_up"]):
            continue
        if pd.isna(row["rs"]) or row["rs"] < MIN_RS_MULTIYEAR:
            continue
        g = _symbol_frame(big, sym)
        rec = pivot_engine.find_long_base(g["DATE"].tolist(), g["HIGH_PRICE"].to_numpy(),
                                           g["LOW_PRICE"].to_numpy(), g["CLOSE_PRICE"].to_numpy(),
                                           last_date.to_pydatetime())
        if rec is None:
            continue
        hits.append(_finish_hit(rec, sym, row, g))
    return hits


def run_vcp(big, uni_df, last_date):
    hits = []
    for sym, row in uni_df.iterrows():
        if not row["above_200dma"]:
            continue
        if pd.isna(row["rs"]) or row["rs"] < MIN_RS_VCP:
            continue
        g = _symbol_frame(big, sym)
        rec = pivot_engine.find_vcp(g["DATE"].tolist(), g["HIGH_PRICE"].to_numpy(),
                                     g["LOW_PRICE"].to_numpy(), g["CLOSE_PRICE"].to_numpy(),
                                     last_date.to_pydatetime())
        if rec is None:
            continue
        hits.append(_finish_hit(rec, sym, row, g))
    return hits


def run_bluesky(big, uni_df, last_date):
    hits = []
    for sym, row in uni_df.iterrows():
        if pd.isna(row["rs"]) or row["rs"] < MIN_RS_BLUESKY:
            continue
        g = _symbol_frame(big, sym)
        rec = pivot_engine.find_blue_sky(g["DATE"].tolist(), g["HIGH_PRICE"].to_numpy(),
                                          g["LOW_PRICE"].to_numpy(), g["CLOSE_PRICE"].to_numpy(),
                                          last_date.to_pydatetime())
        if rec is None:
            continue
        hits.append(_finish_hit(rec, sym, row, g))
    return hits


def run_ipobase(big, uni_df, last_date):
    young = uni_df[(uni_df["first_date"] > IPO_CACHE_START_BUFFER) &
                    (uni_df["first_date"] >= last_date - pd.Timedelta(days=IPO_MAX_AGE_DAYS))]
    hits = []
    for sym, row in young.iterrows():
        g = _symbol_frame(big, sym)
        rec = pivot_engine.find_ipo_base(g["DATE"].tolist(), g["HIGH_PRICE"].to_numpy(),
                                          g["LOW_PRICE"].to_numpy(), g["CLOSE_PRICE"].to_numpy(),
                                          last_date.to_pydatetime())
        if rec is None:
            continue
        hits.append(_finish_hit(rec, sym, row, g))
    return hits


def run_poleflag(big, uni_df, last_date):
    hits = []
    for sym, row in uni_df.iterrows():
        g = _symbol_frame(big, sym)
        rec = pivot_engine.find_pole_flag(g["DATE"].tolist(), g["CLOSE_PRICE"].to_numpy(),
                                           g["TTL_TRD_QNTY"].to_numpy(), last_date.to_pydatetime())
        if rec is None:
            continue
        hits.append(_finish_hit(rec, sym, row, g))
    return hits


def run_dma(big, uni_df, last_date, window):
    hits = []
    for sym, row in uni_df.iterrows():
        g = _symbol_frame(big, sym)
        rec = pivot_engine.find_dma_breakout(g["DATE"].tolist(), g["CLOSE_PRICE"].to_numpy(),
                                              last_date.to_pydatetime(), window)
        if rec is None:
            continue
        # the actual moving MA curve (ma50/ma200/ma_extra) tells this screen's
        # story better than a flat line frozen at the breakout-day SMA value
        hits.append(_finish_hit(rec, sym, row, g, price_pivot=None))
    return hits


def run_record_profit(big, uni_df, last_date):
    hits = record_screen.find_records(list(uni_df.index))
    finished = []
    for rec in hits:
        sym = rec["sym"]
        row = uni_df.loc[sym]
        g = _symbol_frame(big, sym)
        rec["chart_from_idx"] = max(0, len(g) - record_screen.CHART_WINDOW_SESSIONS)
        finished.append(_finish_hit(rec, sym, row, g, price_pivot=None))
    return finished


def run_52wk(big, uni_df, last_date):
    hits = []
    for sym, row in uni_df.iterrows():
        g = _symbol_frame(big, sym)
        rec = pivot_engine.find_52w_high(g["DATE"].tolist(), g["HIGH_PRICE"].to_numpy(),
                                          g["LOW_PRICE"].to_numpy(), g["CLOSE_PRICE"].to_numpy(),
                                          last_date.to_pydatetime())
        if rec is None:
            continue
        hits.append(_finish_hit(rec, sym, row, g))
    return hits


STAGE_PRIORITY = {"fresh_breakout": 0, "both": 0, "climbing": 1, "profit_only": 1, "revenue_only": 1,
                   "forming": 2, "played_out": 3}


def compute_changes(screens_data, prev_path):
    """Diffs today's stage buckets against the most recent prior snapshot,
    per screen: which symbols are newly in the screen, which moved between
    stages, which dropped out entirely. Skips any screen absent from the
    prior snapshot (nothing meaningful to diff — e.g. a screen added since)."""
    if prev_path is None or not prev_path.exists():
        return {}
    with open(prev_path) as f:
        prev = json.load(f)

    changes = {}
    for screen_key, buckets in screens_data["screens"].items():
        prev_buckets = prev.get("screens", {}).get(screen_key)
        if prev_buckets is None:
            continue

        old_map = {r["sym"]: stage for stage, rows in prev_buckets.items() for r in rows}
        new_map, new_data = {}, {}
        for stage, rows in buckets.items():
            for r in rows:
                new_map[r["sym"]] = stage
                new_data[r["sym"]] = r

        events = []
        for sym, stage in new_map.items():
            r = new_data[sym]
            if sym not in old_map:
                events.append({"type": "entered", "sym": sym, "to": stage,
                                "pct_from_pivot": r.get("pct_from_pivot"), "rs": r.get("rs")})
            elif old_map[sym] != stage:
                events.append({"type": "moved", "sym": sym, "from": old_map[sym], "to": stage,
                                "pct_from_pivot": r.get("pct_from_pivot"), "rs": r.get("rs")})
        for sym, stage in old_map.items():
            if sym not in new_map:
                events.append({"type": "left", "sym": sym, "from": stage})

        def sort_key(e):
            type_prio = {"entered": 0, "moved": 1, "left": 2}[e["type"]]
            stage_prio = STAGE_PRIORITY.get(e.get("to") or e.get("from"), 4)
            return (type_prio, stage_prio)

        events.sort(key=sort_key)
        changes[screen_key] = events[:40]
    return changes


def main():
    print("Loading universe...")
    big, qualified, last_date = sc.load_universe()
    print(f"  data through {last_date.date()}, {big['SYMBOL'].nunique()} EQ symbols, {len(qualified)} pass liquidity/mcap")

    print("Computing RS rating + 200-DMA...")
    uni_df = sc.build_universe_frame(big, qualified).set_index("sym")

    screens_data = {"generated_for_date": last_date.strftime("%Y-%m-%d"),
                     "universe_size": len(qualified), "screens": {}}

    runners = [
        ("multiyear", lambda: run_multiyear(big, uni_df, last_date)),
        ("vcp", lambda: run_vcp(big, uni_df, last_date)),
        ("bluesky", lambda: run_bluesky(big, uni_df, last_date)),
        ("ipobase", lambda: run_ipobase(big, uni_df, last_date)),
        ("poleflag", lambda: run_poleflag(big, uni_df, last_date)),
        ("dma20", lambda: run_dma(big, uni_df, last_date, 20)),
        ("dma50", lambda: run_dma(big, uni_df, last_date, 50)),
        ("dma200", lambda: run_dma(big, uni_df, last_date, 200)),
        ("wk52", lambda: run_52wk(big, uni_df, last_date)),
        ("recordprofit", lambda: run_record_profit(big, uni_df, last_date)),
    ]
    bucketers = {"recordprofit": bucket_by_record_type}
    for key, runner in runners:
        print(f"Running {key} screen...")
        hits = runner()
        buckets = bucketers.get(key, bucket_by_stage)(hits)
        for stage, rows in buckets.items():
            print(f"  {stage}: {len(rows)}")
        charts = split_charts(buckets)
        screens_data["screens"][key] = buckets

        charts_path = HERE / f"charts_{key}.json"
        with open(charts_path, "w") as f:
            json.dump(charts, f, separators=(",", ":"))
        print(f"  charts -> {charts_path.name} ({charts_path.stat().st_size / 1e6:.2f} MB)")

    date_str = last_date.strftime("%Y-%m-%d")
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    prev_snapshots = sorted(p for p in SNAPSHOT_DIR.glob("*.json") if p.stem < date_str)
    prev_path = prev_snapshots[-1] if prev_snapshots else None
    screens_data["changes"] = compute_changes(screens_data, prev_path)
    screens_data["changes_since"] = prev_path.stem if prev_path else None
    if prev_path:
        total_changes = sum(len(v) for v in screens_data["changes"].values())
        print(f"Computed changes vs {prev_path.stem}: {total_changes} events across {len(screens_data['changes'])} screens")
    else:
        print("No prior snapshot found — skipping 'what changed' diff")

    out_path = HERE / "screens_data.json"
    with open(out_path, "w") as f:
        json.dump(screens_data, f, indent=1)
    print(f"Wrote {out_path}")

    snap_path = SNAPSHOT_DIR / f"{date_str}.json"
    with open(snap_path, "w") as f:
        json.dump(screens_data, f)
    print(f"Saved snapshot {snap_path}")


if __name__ == "__main__":
    main()
