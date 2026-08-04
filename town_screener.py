"""Which Massachusetts town should you buy a rental in?

The deal simulator answers "is *this* house a good buy". It cannot tell you
where to look in the first place. This does.

For every MA town where we have both a home value and a market rent, it works
out four things an investor actually cares about:

  1. Rent yield      — a year of rent divided by the price. The cash-flow engine.
  2. Yield direction — is that yield better or worse than the town's own recent
                       normal? A town whose yield is climbing is one where
                       prices have fallen behind rents. That is the closest
                       honest definition of "underpriced".
  3. Rent growth     — are rents rising here, or flat?
  4. Buyer leverage  — are homes sitting longer and is inventory building?
                       That is negotiating room.

Each is scored against the rest of Massachusetts, then blended. The blend is
deliberately transparent: every component is returned alongside the score, so
a number you disagree with can be argued with rather than taken on faith.

WHAT THIS IS NOT: it is a screen, not an underwrite. Yield here is gross — no
taxes, insurance, vacancy, or management. Massachusetts property tax rates vary
by more than 2x across towns, which is easily the difference between a good and
a bad deal. Use this to pick a shortlist, then run real listings through the
simulator, which does the actual underwriting.

Run it directly for a ranked table:

    python3 town_screener.py
    python3 town_screener.py --county Middlesex
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import market_data as md

# How far back "recent normal" reaches when judging yield direction.
YIELD_LOOKBACK_MONTHS = 36
YOY_MONTHS = 12

# Blend weights. They sum to 1.0. Rent yield dominates because for a rental,
# cash flow is the thing that keeps you solvent long enough for anything else
# to matter.
WEIGHTS = {
    "yield_z": 0.40,          # how much rent you get per dollar of price
    "yield_trend_z": 0.25,    # is that improving vs the town's own history
    "rent_growth_z": 0.20,    # are rents going up
    "leverage_z": 0.15,       # can you negotiate
}

# A town needs a real market, not three listings and a rumour.
MIN_HOME_VALUE = 100_000


# ---------------------------------------------------------------- helpers
def _zscore(s: pd.Series) -> pd.Series:
    """Standardise within Massachusetts.

    Missingness is preserved: a town with no value for a measure comes back
    NaN, never 0. The scoring step counts NaNs to work out how much of the
    signal it actually has, so quietly turning "unknown" into "average" here
    would inflate every confidence figure downstream.

    A measure where every town is identical carries no information, so present
    values collapse to 0 — average — rather than dividing by zero.
    """
    clean = pd.to_numeric(s, errors="coerce")
    sd = clean.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return clean.where(clean.isna(), 0.0)
    return (clean - clean.mean()) / sd


def _pct_change(latest: pd.Series, prior: pd.Series) -> pd.Series:
    latest = pd.to_numeric(latest, errors="coerce")
    prior = pd.to_numeric(prior, errors="coerce")
    return (latest - prior) / prior.replace(0, np.nan)


# ---------------------------------------------------------------- screen
def build_table() -> pd.DataFrame:
    """Join the four sources into one row per town with the raw measures."""
    zhvi = md.load("zhvi")
    zori = md.load("zori")

    # Home values: level now, and a year ago, and three years ago.
    v_yoy = md.latest_and_lag(zhvi, YOY_MONTHS)
    v_hist = md.latest_and_lag(zhvi, YIELD_LOOKBACK_MONTHS)

    # Market rents: same treatment.
    r_yoy = md.latest_and_lag(zori, YOY_MONTHS)
    r_hist = md.latest_and_lag(zori, YIELD_LOOKBACK_MONTHS)

    t = v_yoy.rename(columns={"latest": "home_value", "prior": "home_value_1y"})
    t = t[["RegionName", "CountyName", "Metro", "home_value", "home_value_1y", "month"]]

    t = t.merge(
        v_hist[["RegionName", "prior"]].rename(columns={"prior": "home_value_3y"}),
        on="RegionName", how="left")
    t = t.merge(
        r_yoy[["RegionName", "latest", "prior"]].rename(
            columns={"latest": "market_rent", "prior": "market_rent_1y"}),
        on="RegionName", how="inner")          # inner: no rent, no screen
    t = t.merge(
        r_hist[["RegionName", "prior"]].rename(columns={"prior": "market_rent_3y"}),
        on="RegionName", how="left")

    # Optional signals — some towns are missing these entirely, which is fine.
    for name, col in (("days_to_pending", "days_pending"), ("inventory", "inventory")):
        try:
            src = md.latest_and_lag(md.load(name), YOY_MONTHS)
        except Exception:                      # noqa: BLE001 - optional signal
            t[col] = np.nan
            t[f"{col}_1y"] = np.nan
            continue
        t = t.merge(
            src[["RegionName", "latest", "prior"]].rename(
                columns={"latest": col, "prior": f"{col}_1y"}),
            on="RegionName", how="left")

    t = t.rename(columns={"RegionName": "town", "CountyName": "county",
                          "Metro": "metro"})
    t = t[t["home_value"] >= MIN_HOME_VALUE]
    return t.dropna(subset=["home_value", "market_rent"]).reset_index(drop=True)


def add_measures(t: pd.DataFrame) -> pd.DataFrame:
    """Turn raw levels into the four things we actually judge on."""
    t = t.copy()

    # 1. Rent yield — a year of rent over the price. Gross.
    t["rent_yield"] = (t["market_rent"] * 12) / t["home_value"]

    # 2. Yield direction — today's yield against the same town three years ago.
    hist_yield = (t["market_rent_3y"] * 12) / t["home_value_3y"]
    t["rent_yield_3y"] = hist_yield
    t["yield_trend"] = t["rent_yield"] - hist_yield

    # 3. Growth rates.
    t["rent_yoy"] = _pct_change(t["market_rent"], t["market_rent_1y"])
    t["price_yoy"] = _pct_change(t["home_value"], t["home_value_1y"])

    # 4. Buyer leverage: homes taking longer to go under agreement, and more of
    #    them on the market, both hand the buyer negotiating power.
    t["days_pending_yoy"] = _pct_change(t["days_pending"], t["days_pending_1y"])
    t["inventory_yoy"] = _pct_change(t["inventory"], t["inventory_1y"])
    return t


def add_score(t: pd.DataFrame) -> pd.DataFrame:
    """Score each town against the rest of Massachusetts.

    Two separate jobs here, kept separate on purpose.

    The score answers "how good does this town look, given what we can see" —
    a weighted mean over the components that exist, renormalised so a town is
    never punished in the *estimate* for data Zillow simply doesn't publish.

    Confidence answers "how much of the signal did we actually have", as a
    share of total weight. It is not folded into the score, because the two
    say different things and averaging them muddles both. Instead `screen()`
    gates on it: a town measured on too little is dropped rather than
    silently ranked as though it were fully observed.

    (Worth recording, since it is not obvious: folding confidence in by
    multiplying the renormalised mean back by available weight looks like
    shrinkage but cancels exactly — it returns the plain fillna-zero sum,
    i.e. treating every missing component as average. The gate is what
    actually does the work.)
    """
    t = t.copy()

    t["yield_z"] = _zscore(t["rent_yield"])
    t["yield_trend_z"] = _zscore(t["yield_trend"])
    t["rent_growth_z"] = _zscore(t["rent_yoy"])

    # Leverage blends the two supply-side signals; either alone is noisy.
    # Stays NaN when neither is present, so it counts as missing below.
    t["leverage_z"] = pd.concat([_zscore(t["days_pending_yoy"]),
                                 _zscore(t["inventory_yoy"])], axis=1).mean(axis=1)

    weighted = sum(t[c].fillna(0.0) * w for c, w in WEIGHTS.items())
    available = sum(t[c].notna() * w for c, w in WEIGHTS.items())
    t["confidence"] = available.round(2)

    t["raw_score"] = (weighted / available.replace(0, np.nan)).fillna(0.0)
    return rescale(t)


def rescale(t: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw blend into a friendly 0-100 rating.

    Applied *after* filtering, not before, so the rating always describes the
    towns actually on screen. Scale it first and the gate then removes the
    top scorers, leaving a shortlist whose best town reads 57 out of 100 for
    no reason a user could follow.
    """
    t = t.copy()
    if t.empty:
        t["score"] = []
        return t

    lo, hi = t["raw_score"].min(), t["raw_score"].max()
    spread = hi - lo
    t["score"] = 50.0 if spread == 0 else (
        (t["raw_score"] - lo) / spread * 100).round(1)
    return t


def plain_read(row: pd.Series) -> str:
    """One sentence a client could read without a finance degree."""
    bits = []

    y = row["rent_yield"]
    if y >= 0.070:
        bits.append("rents are high for the price here")
    elif y >= 0.055:
        bits.append("rent covers a decent share of the price")
    else:
        bits.append("rents are low relative to price")

    if pd.notna(row["yield_trend"]):
        if row["yield_trend"] > 0.004:
            bits.append("and it has been getting better, so prices have fallen "
                        "behind rents")
        elif row["yield_trend"] < -0.004:
            bits.append("and it has been getting worse as prices ran ahead of rents")

    if pd.notna(row["rent_yoy"]):
        if row["rent_yoy"] >= 0.04:
            bits.append(f"rents are up {row['rent_yoy']*100:.0f}% in a year")
        elif row["rent_yoy"] <= 0:
            bits.append("rents are flat or falling")

    if pd.notna(row["days_pending"]) and row["days_pending"] >= 30:
        bits.append(f"homes take about {row['days_pending']:.0f} days to go under "
                    "agreement, so there is room to negotiate")

    return ("; ".join(bits) + ".").capitalize()


def screen(county: str = None, metro: str = None, min_rent_yield: float = None,
           min_confidence: float = 0.6) -> pd.DataFrame:
    """The whole pipeline. Returns towns ranked best-first.

    min_confidence drops towns measured on too little of the signal to rank
    honestly. Pass 0.0 to see everything, thin data included.
    """
    t = add_score(add_measures(build_table()))

    if min_confidence:
        t = t[t["confidence"] >= min_confidence]
    if county:
        t = t[t["county"].str.contains(county, case=False, na=False)]
    if metro:
        t = t[t["metro"].fillna("").str.contains(metro, case=False, na=False)]
    if min_rent_yield is not None:
        t = t[t["rent_yield"] >= min_rent_yield]

    t = rescale(t).sort_values("score", ascending=False).reset_index(drop=True)
    if not t.empty:
        t["read"] = t.apply(plain_read, axis=1)
    return t


def yield_history(town: str, months: int = 60) -> pd.DataFrame:
    """Rent yield month by month for one town — a single series, for charting.

    Deliberately one line and one column of values: the app's charts have to
    stay single-layer.
    """
    values = md.history(md.load("zhvi"), town, months)
    rents = md.history(md.load("zori"), town, months)
    if values.empty or rents.empty:
        return pd.DataFrame(columns=["month", "rent_yield"])

    merged = values.merge(rents, on="month", suffixes=("_value", "_rent"))
    merged["rent_yield"] = (merged["value_rent"] * 12) / merged["value_value"]
    return merged[["month", "rent_yield"]].dropna()


DISPLAY_COLS = ["town", "county", "home_value", "market_rent", "rent_yield",
                "yield_trend", "rent_yoy", "price_yoy", "days_pending",
                "confidence", "score"]


# ---------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description="Rank MA towns for rental investing")
    ap.add_argument("--county")
    ap.add_argument("--metro")
    ap.add_argument("--min-yield", type=float,
                    help="e.g. 0.06 for a 6%% gross rent yield floor")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    t = screen(county=args.county, metro=args.metro, min_rent_yield=args.min_yield)
    if t.empty:
        print("No towns matched.")
        return 1

    show = t.head(args.top).copy()
    show["home_value"] = show["home_value"].map(lambda v: f"${v:,.0f}")
    show["market_rent"] = show["market_rent"].map(lambda v: f"${v:,.0f}")
    for c in ("rent_yield", "yield_trend", "rent_yoy", "price_yoy"):
        show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{v*100:+.1f}%")
    show["days_pending"] = show["days_pending"].map(
        lambda v: "—" if pd.isna(v) else f"{v:.0f}")

    print(f"\n{len(t)} Massachusetts towns screened "
          f"(data through {t['month'].iloc[0]})\n")
    print(show[DISPLAY_COLS].to_string(index=False))
    print(f"\nTop pick — {t['town'].iloc[0]}: {t['read'].iloc[0]}")
    print("\nGross yield only. Run real listings through the simulator to "
          "underwrite taxes, vacancy and financing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
