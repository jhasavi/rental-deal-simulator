"""Massachusetts market data — download, cache, and reshape.

Every source here is free, public, and requires no key or login:

  * Zillow Research public CSVs  (home values, market rents, inventory, speed)
    https://www.zillow.com/research/data/  — published for public use.

Nothing in this module scrapes a website or touches a licensed MLS feed. That
matters: the numbers below can be shown to a client without a data-licence
argument.

The Zillow city files cover the whole country and run to ~90 MB, so we stream
them in chunks and keep only Massachusetts rows. What lands on disk is a few
hundred KB per source.

Run it directly to refresh the cache:

    python3 market_data.py
"""
from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass

import pandas as pd
import requests

# ---------------------------------------------------------------- sources
BASE = "https://files.zillowstatic.com/research/public_csvs"

SOURCES = {
    # typical home value, all homes, smoothed + seasonally adjusted
    "zhvi": f"{BASE}/zhvi/City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    # observed market rent (asking rents actually achieved), all home types
    "zori": f"{BASE}/zori/City_zori_uc_sfrcondomfr_sm_month.csv",
    # for-sale inventory
    "inventory": f"{BASE}/invt_fs/City_invt_fs_uc_sfrcondo_sm_month.csv",
    # median days to pending — how fast the market clears
    "days_to_pending": f"{BASE}/med_doz_pending/City_med_doz_pending_uc_sfrcondo_sm_month.csv",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "market")
STATE = "MA"

# Zillow's identifying columns; everything else is a YYYY-MM-DD observation.
ID_COLS = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName",
           "State", "Metro", "CountyName"]


@dataclass
class Refresh:
    """What one refresh actually did, so callers can report it honestly."""
    name: str
    rows: int
    latest_month: str
    from_cache: bool


# ---------------------------------------------------------------- download
def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}_{STATE}.csv")


def _stream_state_rows(url: str, chunksize: int = 20_000) -> pd.DataFrame:
    """Pull a national Zillow CSV and keep only this state's rows.

    Streaming matters — the city-level home-value file is ~90 MB and we want
    maybe 250 rows of it.
    """
    resp = requests.get(url, stream=True, timeout=180)
    resp.raise_for_status()

    keep = []
    reader = pd.read_csv(io.BytesIO(resp.content), chunksize=chunksize,
                         low_memory=False)
    for chunk in reader:
        if "State" not in chunk.columns:
            raise ValueError(f"unexpected schema from {url}: no State column")
        keep.append(chunk[chunk["State"] == STATE])

    if not keep:
        return pd.DataFrame()
    return pd.concat(keep, ignore_index=True)


def refresh(name: str, force: bool = False) -> Refresh:
    """Download one source (or reuse cache) and return what happened."""
    path = _cache_path(name)
    if os.path.exists(path) and not force:
        df = pd.read_csv(path, low_memory=False)
        return Refresh(name, len(df), _latest_month(df), from_cache=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df = _stream_state_rows(SOURCES[name])
    if df.empty:
        raise ValueError(f"{name}: no {STATE} rows found")
    df.to_csv(path, index=False)
    return Refresh(name, len(df), _latest_month(df), from_cache=False)


def refresh_all(force: bool = False) -> list:
    return [refresh(n, force=force) for n in SOURCES]


# ---------------------------------------------------------------- reshape
def month_cols(df: pd.DataFrame) -> list:
    """The observation columns, oldest first."""
    return sorted(c for c in df.columns if _is_month(c))


def _is_month(col: str) -> bool:
    parts = str(col).split("-")
    return len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 4


def _latest_month(df: pd.DataFrame) -> str:
    cols = month_cols(df)
    return cols[-1] if cols else "n/a"


def load(name: str) -> pd.DataFrame:
    """Load a cached source, downloading it first if we've never fetched it."""
    path = _cache_path(name)
    if not os.path.exists(path):
        refresh(name)
    return pd.read_csv(path, low_memory=False)


def latest_and_lag(df: pd.DataFrame, months_back: int) -> pd.DataFrame:
    """Reduce a wide time series to: town, latest value, value N months ago.

    Returns columns: RegionName, CountyName, Metro, latest, prior, month.
    """
    cols = month_cols(df)
    if not cols:
        raise ValueError("no month columns present")

    latest_col = cols[-1]
    idx = max(0, len(cols) - 1 - months_back)
    prior_col = cols[idx]

    out = df[["RegionName", "CountyName", "Metro"]].copy()
    out["latest"] = pd.to_numeric(df[latest_col], errors="coerce")
    out["prior"] = pd.to_numeric(df[prior_col], errors="coerce")
    out["month"] = latest_col
    out["prior_month"] = prior_col
    return out


def history(df: pd.DataFrame, town: str, months: int = 60) -> pd.DataFrame:
    """Long-format recent history for one town — for charting.

    Long format on purpose: the app's charts must stay single-layer.
    """
    row = df[df["RegionName"] == town]
    if row.empty:
        return pd.DataFrame(columns=["month", "value"])

    cols = month_cols(df)[-months:]
    vals = pd.to_numeric(row.iloc[0][cols], errors="coerce")
    return pd.DataFrame({
        "month": pd.to_datetime(cols),
        "value": vals.values,
    }).dropna()


# ---------------------------------------------------------------- cli
def main() -> int:
    force = "--force" in sys.argv
    print(f"Refreshing Massachusetts market data (force={force})\n")
    try:
        results = refresh_all(force=force)
    except Exception as exc:                      # noqa: BLE001 - surfaced to user
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    for r in results:
        origin = "cached" if r.from_cache else "downloaded"
        print(f"  {r.name:<16} {r.rows:>4} MA towns   latest {r.latest_month}  ({origin})")
    print(f"\nCache: {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
