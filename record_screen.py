"""Record Profit & Revenue screen — stocks whose latest reported quarter is a
new high (within screener.in's visible ~12-quarter/~3-year window) for Net
Profit and/or Sales.

Unlike every other screen in this suite, this one needs a live scrape per
symbol (screener.in has no bulk/bhavcopy-style export for P&L data) — reuses
the existing C:\\home\\ubuntu\\screener_in.py scraper (already used by
oneil_india_bot.py's CAN SLIM fundamentals) and follows the same 14-day cache
TTL convention as fundamentals.py, since quarterly results don't change
between earnings releases.

Caveat, surfaced to the UI too: "record" here means the highest value in
screener.in's visible quarterly table (~12 quarters), not literally since
listing — a company's true all-time-high quarter could sit further back than
that window.
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, r"C:\home\ubuntu")
import screener_in  # noqa: E402

CACHE_PATH = Path(__file__).parent / "records_cache.json"
CACHE_TTL_DAYS = 14
FETCH_DELAY_SEC = 1.0  # matches screener_in.py's own "polite usage: 1 req/sec" docstring
CHART_WINDOW_SESSIONS = 180  # no natural "pivot date" here (this is a fundamentals screen,
                              # not a price-pattern one) — just show ~9 months of recent price


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, default=str))


def _is_fresh(entry: dict) -> bool:
    try:
        ts = datetime.fromisoformat(entry["timestamp"])
        return datetime.now() - ts < timedelta(days=CACHE_TTL_DAYS)
    except Exception:
        return False


SAVE_EVERY = 50  # flush the cache periodically during a long scrape — a ~20min
                  # run that dies partway through (network blip, interrupt) would
                  # otherwise lose every fetch since _save_cache only ran at the end


def _fetch_quarterlies(symbols: list[str]) -> dict[str, dict]:
    """Returns {sym: {quarterly_sales, quarterly_net_profit, quarter_labels}},
    using and updating the on-disk cache."""
    cache = _load_cache()
    out = {}
    fetched = 0
    for i, sym in enumerate(symbols):
        entry = cache.get(sym)
        if entry and _is_fresh(entry):
            out[sym] = entry["data"]
            continue
        data = screener_in.fetch_screener(sym)
        payload = {
            "quarterly_sales": data.get("quarterly_sales", []),
            "quarterly_net_profit": data.get("quarterly_net_profit", []),
            "quarter_labels": data.get("quarter_labels", []),
        }
        cache[sym] = {"timestamp": datetime.now().isoformat(), "data": payload}
        out[sym] = payload
        fetched += 1
        if fetched % SAVE_EVERY == 0:
            _save_cache(cache)
            print(f"  ...{i + 1}/{len(symbols)} symbols processed ({fetched} fresh fetches so far)", flush=True)
        time.sleep(FETCH_DELAY_SEC)
    if fetched:
        _save_cache(cache)
    print(f"  fundamentals: {fetched} fresh scrapes, {len(symbols) - fetched} cache hits")
    return out


def _record_check(series: list):
    """Returns (is_record, latest, prior_best) — None values dropped first."""
    clean = [v for v in series if v is not None]
    if len(clean) < 2:
        return False, None, None
    latest = clean[-1]
    prior_best = max(clean[:-1])
    return latest > prior_best, latest, prior_best


def find_records(symbols: list[str]) -> list[dict]:
    print(f"Fetching fundamentals for {len(symbols)} symbols (screener.in, ~1 req/sec)...")
    data = _fetch_quarterlies(symbols)

    hits = []
    for sym, d in data.items():
        sales = d.get("quarterly_sales") or []
        profit = d.get("quarterly_net_profit") or []
        labels = d.get("quarter_labels") or []

        rev_is_record, rev_latest, rev_prior = _record_check(sales)
        np_is_record, np_latest, np_prior = _record_check(profit)
        if not (rev_is_record or np_is_record):
            continue

        if np_is_record and rev_is_record:
            bucket = "both"
        elif np_is_record:
            bucket = "profit_only"
        else:
            bucket = "revenue_only"

        primary_latest, primary_prior = (np_latest, np_prior) if np_is_record else (rev_latest, rev_prior)
        pct = (primary_latest - primary_prior) / abs(primary_prior) * 100 if primary_prior else None
        quarter_label = labels[-1] if labels else None
        prior_period_np = labels[profit.index(np_prior)] if (np_prior in profit and labels) else None
        prior_period_rev = labels[sales.index(rev_prior)] if (rev_prior in sales and labels) else None

        hits.append({
            "sym": sym,
            "stage": bucket,
            "pivot_date": quarter_label,
            "close": round(primary_latest, 1) if primary_latest is not None else None,
            "pivot": round(primary_prior, 1) if primary_prior is not None else None,
            "pct_from_pivot": round(pct, 1) if pct is not None else 0.0,
            "base_weeks": None,
            "base_depth_pct": None,
            "sessions_since_breakout": None,
            "quarter_label": quarter_label,
            "n_quarters": len([v for v in profit if v is not None]) or len([v for v in sales if v is not None]),
            "net_profit_latest": round(np_latest, 1) if np_latest is not None else None,
            "net_profit_prior_best": round(np_prior, 1) if np_prior is not None else None,
            "net_profit_prior_period": prior_period_np,
            "net_profit_is_record": np_is_record,
            "revenue_latest": round(rev_latest, 1) if rev_latest is not None else None,
            "revenue_prior_best": round(rev_prior, 1) if rev_prior is not None else None,
            "revenue_prior_period": prior_period_rev,
            "revenue_is_record": rev_is_record,
            "chart_from_idx": None,  # filled in by build_screens.py once it knows this symbol's session count
        })
    return hits
