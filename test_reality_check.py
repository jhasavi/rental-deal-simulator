"""Tests for the reality check and the break-even offer price.

These two produce numbers a client may hear out loud in a negotiation, so the
properties worth pinning are the directional ones: cheaper is better, more rent
is better, and the break-even price is genuinely the point where cash flow
crosses zero.

    python3 -m pytest test_reality_check.py -q
"""
import numpy as np
import pandas as pd
import pytest

from rental_deal_simulator import (Assumptions, DealInputs, breakeven_price,
                                   reality_check, run_simulation,
                                   MA_TYPICAL_INSURANCE_RATE,
                                   MA_TYPICAL_TAX_RATE)


def _row(home_value=500_000, market_rent=2_500, town="Testville"):
    return pd.Series({"town": town, "home_value": home_value,
                      "market_rent": market_rent})


def _year1_monthly(price, rent, rate=0.0675, down=0.25):
    """Reproduce the underwriting the two helpers do, for cross-checking."""
    deal = DealInputs(purchase_price=price, down_payment_pct=down, rate=rate,
                      monthly_rent=rent,
                      taxes_annual=price * MA_TYPICAL_TAX_RATE,
                      insurance_annual=price * MA_TYPICAL_INSURANCE_RATE)
    res = run_simulation(deal, Assumptions(years=3, n_trials=200, seed=11))
    return float(np.median(res["annual_cf"][:, 0])) / 12.0


# ---------------------------------------------------------------- reality check
def test_expensive_house_on_thin_rent_does_not_cash_flow():
    cash, verdict = reality_check(_row(home_value=900_000, market_rent=2_000))
    assert cash < 0
    assert "does **not** cash flow" in verdict


def test_cheap_house_on_strong_rent_does_cash_flow():
    cash, verdict = reality_check(_row(home_value=150_000, market_rent=3_000))
    assert cash > 0
    assert "clears about" in verdict


def test_verdict_names_the_town():
    _, verdict = reality_check(_row(town="Brockton"))
    assert "Brockton" in verdict


def test_missing_inputs_return_none():
    assert reality_check(_row(home_value=np.nan)) is None
    assert reality_check(_row(market_rent=np.nan)) is None


def test_more_rent_never_hurts_cash_flow():
    low, _ = reality_check(_row(market_rent=2_000))
    high, _ = reality_check(_row(market_rent=3_500))
    assert high > low


def test_higher_price_never_helps_cash_flow():
    cheap, _ = reality_check(_row(home_value=350_000))
    dear, _ = reality_check(_row(home_value=750_000))
    assert cheap > dear


# ---------------------------------------------------------------- break-even
def test_breakeven_is_actually_the_crossing_point():
    """Just below it the year carries itself; well above it, it does not."""
    row = _row(home_value=600_000, market_rent=2_800)
    be = breakeven_price(row)
    assert be is not None

    assert _year1_monthly(be * 0.97, 2_800) > 0
    assert _year1_monthly(be * 1.15, 2_800) < 0


def test_breakeven_sits_below_a_price_that_does_not_work():
    row = _row(home_value=600_000, market_rent=2_800)
    assert breakeven_price(row) < 600_000


def test_breakeven_returns_typical_price_when_deal_already_works():
    """No point quoting an offer ceiling above what the house costs."""
    row = _row(home_value=200_000, market_rent=3_200)
    assert breakeven_price(row) == pytest.approx(200_000)


def test_breakeven_is_none_when_rent_cannot_carry_anything():
    assert breakeven_price(_row(home_value=800_000, market_rent=150)) is None


def test_more_rent_supports_a_higher_offer():
    low = breakeven_price(_row(home_value=900_000, market_rent=2_500))
    high = breakeven_price(_row(home_value=900_000, market_rent=4_000))
    assert high > low


def test_lower_rate_supports_a_higher_offer():
    dear = breakeven_price(_row(home_value=900_000, market_rent=3_000),
                           rate=0.085)
    cheap = breakeven_price(_row(home_value=900_000, market_rent=3_000),
                            rate=0.050)
    assert cheap > dear


def test_breakeven_handles_missing_data():
    assert breakeven_price(_row(home_value=np.nan)) is None
