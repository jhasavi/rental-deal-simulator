"""Tests for the town screener.

The scoring logic decides where money gets pointed, so the properties that
matter are pinned here: yield arithmetic, and the rule that thin data must not
outrank thick data.

    python3 -m pytest test_town_screener.py -q
"""
import numpy as np
import pandas as pd
import pytest

import town_screener as ts


def _town(name, home_value=500_000, market_rent=2_500, home_value_1y=490_000,
          market_rent_1y=2_400, home_value_3y=450_000, market_rent_3y=2_100,
          days_pending=20, days_pending_1y=18,
          inventory=100, inventory_1y=90):
    return {
        "town": name, "county": "Test County", "metro": "Boston",
        "month": "2026-06-30",
        "home_value": home_value, "home_value_1y": home_value_1y,
        "home_value_3y": home_value_3y,
        "market_rent": market_rent, "market_rent_1y": market_rent_1y,
        "market_rent_3y": market_rent_3y,
        "days_pending": days_pending, "days_pending_1y": days_pending_1y,
        "inventory": inventory, "inventory_1y": inventory_1y,
    }


def _frame(rows):
    return ts.add_measures(pd.DataFrame(rows))


# ---------------------------------------------------------------- measures
def test_rent_yield_is_annual_rent_over_price():
    t = _frame([_town("A", home_value=600_000, market_rent=3_000)])
    assert t["rent_yield"].iloc[0] == pytest.approx(36_000 / 600_000)


def test_yield_trend_positive_when_prices_fall_behind_rents():
    """Rents up 20%, prices flat -> yield improved. This is the 'underpriced'
    signal, so its sign must be right."""
    t = _frame([_town("A", home_value=500_000, home_value_3y=500_000,
                      market_rent=2_400, market_rent_3y=2_000)])
    assert t["yield_trend"].iloc[0] > 0


def test_yield_trend_negative_when_prices_outrun_rents():
    t = _frame([_town("A", home_value=700_000, home_value_3y=500_000,
                      market_rent=2_100, market_rent_3y=2_000)])
    assert t["yield_trend"].iloc[0] < 0


def test_yoy_growth_rates():
    t = _frame([_town("A", market_rent=2_200, market_rent_1y=2_000,
                      home_value=525_000, home_value_1y=500_000)])
    assert t["rent_yoy"].iloc[0] == pytest.approx(0.10)
    assert t["price_yoy"].iloc[0] == pytest.approx(0.05)


def test_zero_prior_does_not_blow_up():
    t = _frame([_town("A", market_rent_1y=0, home_value_1y=0)])
    assert pd.isna(t["rent_yoy"].iloc[0])
    assert pd.isna(t["price_yoy"].iloc[0])


# ---------------------------------------------------------------- scoring
def test_higher_yield_scores_higher_all_else_equal():
    t = ts.add_score(_frame([
        _town("Low", home_value=800_000, market_rent=2_000),
        _town("High", home_value=400_000, market_rent=2_000),
    ]))
    by_town = t.set_index("town")["score"]
    assert by_town["High"] > by_town["Low"]


def test_confidence_is_full_when_every_component_present():
    t = ts.add_score(_frame([_town("A"), _town("B", home_value=400_000)]))
    assert (t["confidence"] == 1.0).all()


def test_missing_components_reduce_confidence():
    thin = _town("Thin")
    thin.update(market_rent_3y=np.nan, home_value_3y=np.nan,   # no yield trend
                market_rent_1y=np.nan,                          # no rent growth
                days_pending=np.nan, days_pending_1y=np.nan,
                inventory=np.nan, inventory_1y=np.nan)          # no leverage
    t = ts.add_score(_frame([thin, _town("Full")]))
    conf = t.set_index("town")["confidence"]
    assert conf["Thin"] < conf["Full"]
    assert conf["Thin"] == pytest.approx(ts.WEIGHTS["yield_z"])


def _yield_only(name, home_value, market_rent):
    """A town where rent yield is the only measurable component."""
    t = _town(name, home_value=home_value, market_rent=market_rent)
    t.update(market_rent_3y=np.nan, home_value_3y=np.nan,
             market_rent_1y=np.nan,
             days_pending=np.nan, days_pending_1y=np.nan,
             inventory=np.nan, inventory_1y=np.nan)
    return t


def test_thin_towns_are_scored_on_what_they_have():
    """With yield the only observable, ranking must follow yield exactly.

    This pins the renormalisation: if absent components were folded in as
    average, they would dilute the one real signal and the order could drift.
    """
    towns = [_yield_only("Best", 300_000, 2_400),
             _yield_only("Mid", 500_000, 2_400),
             _yield_only("Worst", 900_000, 2_400)]
    t = ts.add_score(_frame(towns)).sort_values("score", ascending=False)

    assert list(t["town"]) == ["Best", "Mid", "Worst"]
    assert t["confidence"].tolist() == pytest.approx(
        [ts.WEIGHTS["yield_z"]] * 3)


def test_thin_data_is_gated_out_before_it_can_mislead():
    """The bug this guards: a town known only for an extreme yield topping the
    ranking. It is the confidence gate, not the score, that stops it."""
    thin = _town("ThinButHighYield", home_value=200_000, market_rent=2_500)
    thin.update(market_rent_3y=np.nan, home_value_3y=np.nan,
                market_rent_1y=np.nan,
                days_pending=np.nan, days_pending_1y=np.nan,
                inventory=np.nan, inventory_1y=np.nan)
    thick = _town("ThickAndGood", home_value=400_000, market_rent=2_400,
                  home_value_3y=400_000, market_rent_3y=1_900,
                  market_rent_1y=2_150,
                  days_pending=35, days_pending_1y=20,
                  inventory=150, inventory_1y=100)
    middling = _town("Middling", home_value=700_000, market_rent=2_000)

    t = ts.add_score(_frame([thin, thick, middling]))
    survivors = t[t["confidence"] >= 0.6].sort_values("score", ascending=False)
    assert "ThinButHighYield" not in set(survivors["town"])
    assert survivors["town"].iloc[0] == "ThickAndGood"


def test_screen_filters_out_low_confidence_towns():
    thin = _town("Thin")
    thin.update(market_rent_3y=np.nan, home_value_3y=np.nan,
                market_rent_1y=np.nan,
                days_pending=np.nan, days_pending_1y=np.nan,
                inventory=np.nan, inventory_1y=np.nan)
    t = ts.add_score(_frame([thin, _town("Full")]))
    kept = t[t["confidence"] >= 0.6]
    assert list(kept["town"]) == ["Full"]


def test_scores_span_the_full_range():
    t = ts.add_score(_frame([
        _town("A", home_value=300_000), _town("B", home_value=600_000),
        _town("C", home_value=900_000),
    ]))
    assert t["score"].min() == 0.0
    assert t["score"].max() == 100.0


def test_identical_towns_do_not_produce_nan_scores():
    t = ts.add_score(_frame([_town("A"), _town("B")]))
    assert t["score"].notna().all()


# ---------------------------------------------------------------- wording
def test_plain_read_flags_improving_yield():
    t = _frame([_town("A", home_value=500_000, home_value_3y=500_000,
                      market_rent=2_600, market_rent_3y=2_000)])
    assert "fallen behind" in ts.plain_read(t.iloc[0])


def test_plain_read_is_a_sentence():
    t = ts.add_score(_frame([_town("A"), _town("B", home_value=300_000)]))
    for _, row in t.iterrows():
        read = ts.plain_read(row)
        assert read[0].isupper() and read.endswith(".")
