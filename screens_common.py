"""Shared data engine for the pattern-screens suite (Multi-year breakouts, VCP,
Blue sky, IPO base) — universe filtering, 200-DMA, and a 1-99 percentile RS Rating,
computed once per run and shared across all 4 screens.

Data: C:\\opt\\nse_gapbot\\bhav_cache\\bhav_YYYYMMDD.parquet (daily EQ bhavcopy,
2023-08-14 onward) + mcap_YYYYMMDD.parquet (market cap per symbol).

Gotcha carried over from the existing scanner suite (base_on_base_scanner.py,
daily_base_breakout_scanner.py): TURNOVER_LACS is already in crores despite the
column name — never rescale it.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

BHAV_DIR = Path(r"C:\opt\nse_gapbot\bhav_cache")

ETF_PATTERNS = re.compile(r"BEES$|ETF$|IETF$|GOLD$|SILVER$|LIQUID|GROWTHADD$|NIFTY|SENSEX", re.I)

MIN_MCAP_CR = 500.0
MIN_AVG_TURNOVER_CR = 5.0
MIN_PRICE = 20.0
TURNOVER_WINDOW = 20

RS_QUARTER = 63  # ~1 trading quarter in sessions
RS_LOOKBACK = RS_QUARTER * 4  # ~12 months


def _load_all_bhav() -> pd.DataFrame:
    files = sorted(glob.glob(str(BHAV_DIR / "bhav_*.parquet")))
    if not files:
        raise SystemExit(f"no bhav_*.parquet files in {BHAV_DIR}")
    cols = ["SYMBOL", "SERIES", "DATE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
            "CLOSE_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS"]
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        df = df[df["SERIES"] == "EQ"]
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    big["DATE"] = pd.to_datetime(big["DATE"])
    big = big.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)
    return big


def _latest_mcap() -> pd.DataFrame:
    files = sorted(glob.glob(str(BHAV_DIR / "mcap_*.parquet")))
    if not files:
        raise SystemExit(f"no mcap_*.parquet files in {BHAV_DIR}")
    return pd.read_parquet(files[-1], columns=["SYMBOL", "SERIES", "MCAP_CR"])


def load_universe():
    """Returns (big: full daily history for ALL EQ symbols sorted by SYMBOL,DATE,
    qualified: set of symbols passing the liquidity/mcap/price/ETF filters,
    last_date: the most recent bhav date as a pandas Timestamp)."""
    big = _load_all_bhav()
    last_date = big["DATE"].max()

    mcap = _latest_mcap()
    mcap_ok = set(mcap.loc[(mcap["SERIES"] == "EQ") & (mcap["MCAP_CR"] >= MIN_MCAP_CR), "SYMBOL"])

    qualified = set()
    for sym, g in big.groupby("SYMBOL", sort=False):
        if ETF_PATTERNS.search(sym):
            continue
        if sym not in mcap_ok:
            continue
        if g["DATE"].iloc[-1] != last_date:
            continue  # stale/delisted, not in today's bhav
        if g["CLOSE_PRICE"].iloc[-1] < MIN_PRICE:
            continue
        avg_turnover = g["TURNOVER_LACS"].tail(TURNOVER_WINDOW).mean()
        if avg_turnover < MIN_AVG_TURNOVER_CR:
            continue
        qualified.add(sym)

    return big, qualified, last_date


def compute_200dma(closes: pd.Series):
    """Returns (above_200dma: bool, turning_up: bool, dma_series) for the LATEST bar."""
    if len(closes) < 200:
        return False, False, None
    dma = closes.rolling(200).mean()
    above = bool(closes.iloc[-1] > dma.iloc[-1])
    turning_up = bool(len(dma.dropna()) > 20 and dma.iloc[-1] > dma.iloc[-21])
    return above, turning_up, dma


def compute_rs_raw(closes: pd.Series):
    """IBD-style weighted-quarter score: 0.4*Q1 + 0.2*(Q2+Q3+Q4), Q1 = most recent
    ~63-session return, Q2-Q4 = the three prior ~63-session periods. None if the
    stock doesn't have RS_LOOKBACK sessions of history yet."""
    n = len(closes)
    if n < RS_LOOKBACK + 1:
        return None
    c = closes.values
    idx = [n - 1, n - 1 - RS_QUARTER, n - 1 - 2 * RS_QUARTER, n - 1 - 3 * RS_QUARTER, n - 1 - 4 * RS_QUARTER]
    p0, p1, p2, p3, p4 = (c[i] for i in idx)
    if min(p0, p1, p2, p3, p4) <= 0:
        return None
    q1 = p0 / p1 - 1.0
    q2 = p1 / p2 - 1.0
    q3 = p2 / p3 - 1.0
    q4 = p3 / p4 - 1.0
    return 0.4 * q1 + 0.2 * (q2 + q3 + q4)


def build_universe_frame(big: pd.DataFrame, qualified: set[str]) -> pd.DataFrame:
    """One row per qualified symbol: close, above_200dma, dma_turning_up, rs_raw,
    rs (1-99 percentile, None if rs_raw is None), avg_turnover_cr, first_date
    (for IPO-base universe filtering), full history handle via `big` groupby."""
    rows = []
    for sym in qualified:
        g = big[big["SYMBOL"] == sym]
        closes = g["CLOSE_PRICE"]
        above, turning_up, _ = compute_200dma(closes)
        rs_raw = compute_rs_raw(closes)
        rows.append({
            "sym": sym,
            "close": float(closes.iloc[-1]),
            "above_200dma": above,
            "dma_turning_up": turning_up,
            "rs_raw": rs_raw,
            "avg_turnover_cr": float(g["TURNOVER_LACS"].tail(TURNOVER_WINDOW).mean()),
            "first_date": g["DATE"].iloc[0],
            "n_sessions": len(g),
        })
    df = pd.DataFrame(rows)
    has_rs = df["rs_raw"].notna()
    df["rs"] = np.nan
    df.loc[has_rs, "rs"] = df.loc[has_rs, "rs_raw"].rank(pct=True) * 98 + 1
    df["rs"] = df["rs"].round(0)
    return df
