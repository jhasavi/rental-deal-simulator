"""Tests for the ADU model.

The statutory rules are pinned hard, because getting the size cap backwards
would overstate feasibility on exactly the small houses where it binds — and
several published summaries of this law do get it backwards.

    python3 -m pytest test_adu.py -q
"""
import pytest

import adu


# ---------------------------------------------------------------- the statute
def test_large_house_is_capped_at_the_statutory_900():
    assert adu.max_adu_sqft(4_000) == 900.0


def test_small_house_is_capped_at_half_its_own_area():
    """The cap is the SMALLER of 900 and half the house — so a 1,200 sq ft
    house gets 600, not 900."""
    assert adu.max_adu_sqft(1_200) == 600.0


def test_1800_sqft_is_the_break_point_for_a_full_size_adu():
    assert adu.max_adu_sqft(1_800) == 900.0
    assert adu.max_adu_sqft(1_799) < 900.0


def test_nonsense_house_size_yields_no_adu():
    assert adu.max_adu_sqft(0) == 0.0
    assert adu.max_adu_sqft(-500) == 0.0
    assert adu.max_adu_sqft(None) == 0.0


# ---------------------------------------------------------------- build costs
def test_conversion_is_cheaper_than_detached_new_build():
    """The route matters more than the finishes — reusing a shell skips
    foundation, framing and roof."""
    sqft = 900
    assert (adu.build_cost(sqft, "basement or attic conversion")
            < adu.build_cost(sqft, "detached new build"))


def test_cost_position_walks_the_range():
    lo = adu.build_cost(900, "detached new build", position=0.0)
    mid = adu.build_cost(900, "detached new build", position=0.5)
    hi = adu.build_cost(900, "detached new build", position=1.0)
    assert lo < mid < hi
    assert lo == 900 * adu.BUILD_COSTS["detached new build"][0]
    assert hi == 900 * adu.BUILD_COSTS["detached new build"][1]


def test_unknown_build_type_is_rejected():
    with pytest.raises(ValueError):
        adu.build_cost(900, "hot air balloon")


def test_plan_uses_the_largest_legal_unit():
    plan = adu.plan_for(1_400, "garage conversion")
    assert plan.adu_sqft == 700.0
    assert plan.build_cost == adu.build_cost(700, "garage conversion")


# ---------------------------------------------------------------- economics
def test_more_financing_means_less_cash_but_a_higher_hurdle():
    cheap = adu.marginal_return(
        adu.plan_for(1_800, financed_pct=0.0), 2_800)
    levered = adu.marginal_return(
        adu.plan_for(1_800, financed_pct=0.8), 2_800)

    assert levered["cash_invested"] < cheap["cash_invested"]
    assert levered["breakeven_rent"] > cheap["breakeven_rent"]
    assert levered["monthly_net"] < cheap["monthly_net"]


def test_unlevered_breakeven_only_has_to_cover_tax_and_insurance():
    plan = adu.plan_for(1_800, financed_pct=0.0)
    costs = adu.carrying_cost(plan)
    assert costs["loan_payment"] == 0.0
    assert adu.breakeven_rent(plan) < 500     # nothing but carrying costs


def test_rent_at_breakeven_nets_about_zero():
    plan = adu.plan_for(1_800)
    be = adu.breakeven_rent(plan)
    assert adu.net_monthly(plan, be) == pytest.approx(0.0, abs=1e-6)


def test_above_breakeven_is_positive_and_below_is_negative():
    plan = adu.plan_for(1_800)
    be = adu.breakeven_rent(plan)
    assert adu.net_monthly(plan, be * 1.2) > 0
    assert adu.net_monthly(plan, be * 0.8) < 0


def test_cheaper_build_lowers_the_rent_you_need():
    conv = adu.breakeven_rent(adu.plan_for(1_800, "basement or attic conversion"))
    new = adu.breakeven_rent(adu.plan_for(1_800, "detached new build"))
    assert conv < new


def test_no_cash_invested_reports_no_cash_on_cash():
    """Fully financed means the return on cash is undefined, not infinite."""
    plan = adu.plan_for(1_800, financed_pct=1.0)
    result = adu.marginal_return(plan, 2_800)
    assert result["cash_invested"] == 0
    assert result["cash_on_cash"] is None
    assert result["monthly_net"] > 0          # still cash flows


def test_carrying_cost_includes_the_higher_assessment():
    """Building an ADU raises the assessed value, so the tax bill goes up.
    Forgetting that flatters every one of these deals."""
    plan = adu.plan_for(1_800)
    assert adu.carrying_cost(plan)["added_tax"] > 0


def test_yield_on_cost_is_annual_rent_over_build_cost():
    plan = adu.plan_for(1_800, "basement or attic conversion")
    result = adu.marginal_return(plan, 2_000)
    assert result["yield_on_cost"] == pytest.approx(24_000 / plan.build_cost)


def test_punitive_fees_make_breakeven_impossible():
    plan = adu.plan_for(1_800)
    assert adu.breakeven_rent(plan, vacancy=0.5, management=0.5,
                              maintenance=0.6) is None
