"""Tests for the DSCR loan sizer.

The headline claim this module exists to fix — that the tool's own investor
DSCR convention (1.02 on the default deal) and the lender's actual convention
(1.21) disagree enough to change a fundability verdict — is pinned first.
Everything after it protects the closed-form sizing math: which constraint
binds, monotonicity in the variables a client will actually ask about, and the
edge cases (zero rent, past the prepay window) that a closed form gets wrong
first if it's wrong at all.

    python3 -m pytest test_dscr.py -q
"""
import math

import pytest

import dscr
from dscr import DscrProgram


# ---------------------------------------------------------------- the finding
def test_lender_convention_reads_higher_than_investor_convention():
    """The simulator's default deal: $650k, $4,800/mo rent, 25% down, 6.75%,
    30-year. Investor DSCR (NOI/P&I) is ~1.02 — the tool today calls that a
    lender decline. Lender DSCR (gross rent/PITIA) is ~1.21 — fundable. Both
    numbers are real; conflating them is the bug."""
    price, down_pct, rate, term = 650_000.0, 0.25, 0.0675, 30
    rent_m, taxes, ins = 4_800.0, 7_200.0, 2_400.0
    loan = price * (1 - down_pct)
    pf = dscr.payment_factor(rate, term)
    annual_pi = loan * pf

    lender = dscr.lender_dscr(rent_m, annual_pi, taxes, ins)
    assert lender == pytest.approx(1.212, abs=0.01)
    assert lender > 1.15   # the tool should call this fundable, not a decline


# ---------------------------------------------------------------- payment_factor
def test_interest_only_factor_is_just_the_rate():
    assert dscr.payment_factor(0.08, 30, interest_only=True) == pytest.approx(0.08)


def test_interest_only_has_no_term_dependence():
    assert (dscr.payment_factor(0.08, 15, interest_only=True)
            == dscr.payment_factor(0.08, 30, interest_only=True))


def test_amortizing_factor_exceeds_interest_only_at_same_rate():
    """Amortizing pays down principal too, so it always costs more per year
    than interest-only at the same rate — this is what makes IO the lever
    that raises a DSCR-bound loan."""
    amort = dscr.payment_factor(0.08, 30, interest_only=False)
    io = dscr.payment_factor(0.08, 30, interest_only=True)
    assert amort > io


def test_zero_rate_factor_is_straight_line():
    assert dscr.payment_factor(0.0, 25, interest_only=False) == pytest.approx(1 / 25)


# ---------------------------------------------------------------- lender_dscr
def test_lender_dscr_zero_pitia_is_infinite():
    assert dscr.lender_dscr(3_000, 0.0, 0.0, 0.0) == float("inf")


def test_lender_dscr_matches_hand_arithmetic():
    # rent 3000/mo = 36000/yr; PITIA = 20000 pi + 6000 tax + 2000 ins = 28000
    assert dscr.lender_dscr(3_000, 20_000, 6_000, 2_000) == pytest.approx(36_000 / 28_000)


# ---------------------------------------------------------------- max_loan
def test_dscr_bound_loan_round_trips_to_exactly_min_dscr():
    """When DSCR is the binding constraint, the loan sizing is the inversion
    of the DSCR formula — so pricing that loan back out must return exactly
    the program's min_dscr, not just something close to it."""
    program = DscrProgram(name="test", min_dscr=1.20, max_ltv=0.95, rate=0.08)
    result = dscr.max_loan(price=1_000_000, rent_m=6_000, taxes=8_000,
                           ins=3_000, program=program)
    assert result["binding"] == "dscr"
    assert result["dscr"] == pytest.approx(1.20, abs=1e-6)


def test_ltv_bound_loan_is_capped_at_program_max_ltv():
    """Generous rent with a tight LTV cap: LTV binds, and the loan is exactly
    max_ltv * price regardless of how much rent would otherwise support."""
    program = DscrProgram(name="test", min_dscr=1.0, max_ltv=0.70, rate=0.08)
    result = dscr.max_loan(price=500_000, rent_m=20_000, taxes=2_000,
                           ins=1_000, program=program)
    assert result["binding"] == "ltv"
    assert result["loan"] == pytest.approx(0.70 * 500_000)


def test_more_rent_grows_a_dscr_bound_loan_but_not_an_ltv_bound_one():
    program_tight_dscr = DscrProgram(name="test", min_dscr=1.30, max_ltv=0.90,
                                     rate=0.08)
    low = dscr.max_loan(700_000, 3_000, 8_000, 3_000, program_tight_dscr)
    high = dscr.max_loan(700_000, 6_000, 8_000, 3_000, program_tight_dscr)
    assert low["binding"] == "dscr"
    assert high["loan"] > low["loan"]

    program_tight_ltv = DscrProgram(name="test", min_dscr=0.5, max_ltv=0.60,
                                    rate=0.08)
    low2 = dscr.max_loan(700_000, 3_000, 8_000, 3_000, program_tight_ltv)
    high2 = dscr.max_loan(700_000, 20_000, 8_000, 3_000, program_tight_ltv)
    assert low2["binding"] == "ltv" and high2["binding"] == "ltv"
    assert low2["loan"] == pytest.approx(high2["loan"])


def test_higher_rate_shrinks_a_dscr_bound_loan():
    cheap = DscrProgram(name="cheap", min_dscr=1.25, max_ltv=0.95, rate=0.06)
    pricey = DscrProgram(name="pricey", min_dscr=1.25, max_ltv=0.95, rate=0.10)
    lo = dscr.max_loan(700_000, 4_500, 7_000, 2_500, pricey)
    hi = dscr.max_loan(700_000, 4_500, 7_000, 2_500, cheap)
    assert lo["binding"] == hi["binding"] == "dscr"
    assert hi["loan"] > lo["loan"]


def test_higher_min_dscr_shrinks_the_loan():
    loose = DscrProgram(name="loose", min_dscr=1.0, max_ltv=0.95, rate=0.08)
    strict = DscrProgram(name="strict", min_dscr=1.4, max_ltv=0.95, rate=0.08)
    lo = dscr.max_loan(700_000, 4_500, 7_000, 2_500, strict)
    hi = dscr.max_loan(700_000, 4_500, 7_000, 2_500, loose)
    assert hi["loan"] > lo["loan"]


def test_interest_only_sizes_a_bigger_loan_than_amortizing():
    """Same rent, same DSCR floor: IO's smaller required payment supports a
    bigger loan whenever DSCR (not LTV) binds."""
    amort = DscrProgram(name="amort", min_dscr=1.25, max_ltv=0.95, rate=0.08,
                        interest_only=False)
    io = DscrProgram(name="io", min_dscr=1.25, max_ltv=0.95, rate=0.08,
                     interest_only=True)
    r_amort = dscr.max_loan(700_000, 4_200, 7_000, 2_500, amort)
    r_io = dscr.max_loan(700_000, 4_200, 7_000, 2_500, io)
    assert r_amort["binding"] == r_io["binding"] == "dscr"
    assert r_io["loan"] > r_amort["loan"]


def test_zero_or_negative_rent_yields_zero_loan_no_crash():
    program = DscrProgram(name="test", min_dscr=1.2, max_ltv=0.8, rate=0.08)
    for bad_rent in (0.0, -500.0):
        result = dscr.max_loan(600_000, bad_rent, 6_000, 2_000, program)
        assert result["loan"] == 0.0
        assert result["down_payment"] == 600_000.0


def test_zero_price_yields_zero_loan_no_crash():
    program = DscrProgram(name="test", min_dscr=1.2, max_ltv=0.8, rate=0.08)
    result = dscr.max_loan(0.0, 3_000, 0.0, 0.0, program)
    assert result["loan"] == 0.0
    assert not math.isnan(result["ltv"])


def test_cash_to_close_includes_down_payment_points_and_other_closing():
    program = DscrProgram(name="test", min_dscr=1.0, max_ltv=0.75, rate=0.08,
                          points_pct=0.02, other_closing_pct=0.01)
    result = dscr.max_loan(500_000, 5_000, 6_000, 2_000, program)
    expected = (result["down_payment"] + result["loan"] * 0.02
               + 500_000 * 0.01)
    assert result["cash_to_close"] == pytest.approx(expected)


# ---------------------------------------------------------------- compare / best
def test_compare_programs_preserves_input_order():
    names = [r["program"] for r in dscr.compare_programs(
        700_000, 4_500, 7_000, 2_500)]
    assert names == [p.name for p in dscr.DEFAULT_PROGRAMS]


def test_best_program_picks_the_largest_loan():
    rows = dscr.compare_programs(700_000, 4_500, 7_000, 2_500)
    best = dscr.best_program(700_000, 4_500, 7_000, 2_500)
    assert best["loan"] == max(r["loan"] for r in rows)


# ---------------------------------------------------------------- breakeven_rent
def test_breakeven_rent_round_trips_through_max_loan():
    """The rent breakeven_rent names should, fed back into max_loan, produce
    a loan matching the target LTV — the two functions invert each other."""
    program = DscrProgram(name="test", min_dscr=1.2, max_ltv=0.75, rate=0.075)
    price, target_ltv = 600_000, 0.75
    needed_rent = dscr.breakeven_rent(price, target_ltv, 6_500, 2_200, program)
    result = dscr.max_loan(price, needed_rent, 6_500, 2_200, program)
    assert result["ltv"] == pytest.approx(target_ltv, abs=1e-4)


def test_breakeven_rent_rises_with_target_leverage():
    program = DscrProgram(name="test", min_dscr=1.2, max_ltv=0.90, rate=0.075)
    low_lev = dscr.breakeven_rent(600_000, 0.5, 6_500, 2_200, program)
    high_lev = dscr.breakeven_rent(600_000, 0.85, 6_500, 2_200, program)
    assert high_lev > low_lev


# ---------------------------------------------------------------- prepay_penalty
def test_prepay_penalty_follows_the_step_down_schedule():
    program = DscrProgram(name="test", min_dscr=1.2, max_ltv=0.8, rate=0.08,
                          prepay_step_down=(0.05, 0.04, 0.03, 0.02, 0.01))
    balance = 400_000.0
    assert dscr.prepay_penalty(balance, 1, program) == pytest.approx(400_000 * 0.05)
    assert dscr.prepay_penalty(balance, 3, program) == pytest.approx(400_000 * 0.03)
    assert dscr.prepay_penalty(balance, 5, program) == pytest.approx(400_000 * 0.01)


def test_prepay_penalty_is_zero_past_the_window():
    program = DscrProgram(name="test", min_dscr=1.2, max_ltv=0.8, rate=0.08)
    assert dscr.prepay_penalty(400_000, 6, program) == 0.0
    assert dscr.prepay_penalty(400_000, 100, program) == 0.0


def test_prepay_penalty_rejects_year_zero():
    program = DscrProgram(name="test", min_dscr=1.2, max_ltv=0.8, rate=0.08)
    assert dscr.prepay_penalty(400_000, 0, program) == 0.0
