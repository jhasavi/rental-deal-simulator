"""DSCR loans — sizing the loan a rental-property lender will actually make.

WHY THIS EXISTS. The simulator's headline DSCR number (see `run_simulation` in
rental_deal_simulator.py) answers an investor's question: after vacancy,
management and maintenance, does the rent cover the mortgage? That is honest
accounting, but it is not what a DSCR lender computes, and the gap is not
small.

A DSCR loan qualifies the deal on gross rent over PITIA (principal, interest,
taxes, insurance, association dues) — not on net operating income over P&I. On
the simulator's own default deal ($650,000, $4,800/mo rent, 25% down, 6.75%,
30-year):

    investor convention  (NOI / P&I)     -> 1.02   "most lenders decline"
    lender convention     (rent / PITIA)  -> 1.21   fundable at a 1.15 or 1.00
                                                     program

Both numbers are real. 1.02 is the stricter, all-in test of whether the
property carries itself once you run it like a business. 1.21 is what the
loan officer's worksheet says. Quoting only the investor number tells a client
a financeable deal is not financeable — that is the mistake this module
exists to stop making.

This module also turns DSCR from a pass/fail grade into what it actually is
for a lender: the constraint that sizes the loan. Given a price and a rent, a
DSCR program does not ask "does this pass" — it answers "how much would you
lend", and that number is sometimes set by the DSCR minimum and sometimes by
plain loan-to-value. Knowing *which one binds* tells you whether more rent
would unlock more leverage (DSCR-bound) or whether it wouldn't (LTV-bound).

NOT A LOAN QUOTE. Every real lender has its own overlays, rent-haircut rules,
reserve requirements and pricing. The three programs below are typical market
tiers as of 2026, meant to be edited to match an actual term sheet before
being read aloud to a client.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------- programs
@dataclass
class DscrProgram:
    """One lender's DSCR product. Rate, LTV and the DSCR floor move together —
    looser DSCR and higher LTV cost more, in the usual way."""
    name: str
    min_dscr: float
    max_ltv: float
    rate: float
    term_years: int = 30
    interest_only: bool = False
    io_years: int = 10
    points_pct: float = 0.02          # of the loan amount
    other_closing_pct: float = 0.01   # of purchase price (title, appraisal, etc.)
    prepay_step_down: tuple = (0.05, 0.04, 0.03, 0.02, 0.01)   # 5/4/3/2/1, by year


DEFAULT_PROGRAMS = (
    DscrProgram(name="1.25 min DSCR — best rate", min_dscr=1.25, max_ltv=0.70,
               rate=0.0775),
    DscrProgram(name="1.15 min DSCR — standard", min_dscr=1.15, max_ltv=0.75,
               rate=0.0800),
    DscrProgram(name="1.00 min DSCR — max leverage", min_dscr=1.00, max_ltv=0.80,
               rate=0.0850),
)


# ---------------------------------------------------------------- payment math
def payment_factor(rate: float, term_years: int, interest_only: bool = False) -> float:
    """Annual debt service per $1 of loan.

    Interest-only collapses this to just the rate — no principal, so no term
    dependence — which is exactly what makes IO the lever that raises a
    DSCR-bound loan: same rent, smaller required payment, bigger loan.
    """
    if interest_only:
        return rate
    if rate <= 0:
        return 1.0 / term_years
    r = rate / 12.0
    n = term_years * 12
    return 12.0 * r / (1 - (1 + r) ** -n)


def lender_dscr(rent_m: float, annual_pi: float, taxes: float, ins: float,
                hoa_m: float = 0.0) -> float:
    """Gross scheduled rent over PITIA — the lender's convention, not the
    investor's. `annual_pi` is principal + interest for the year; taxes and
    ins are annual; hoa_m is monthly."""
    pitia = annual_pi + taxes + ins + hoa_m * 12.0
    if pitia <= 0:
        return float("inf")
    return (rent_m * 12.0) / pitia


# ---------------------------------------------------------------- sizing
def max_loan(price: float, rent_m: float, taxes: float, ins: float,
            program: DscrProgram, hoa_m: float = 0.0) -> dict:
    """The loan this program will actually make on this deal.

    Closed form, not a search: the DSCR floor sets an allowance for annual
    P&I (whatever's left of qualifying rent after taxes/insurance/HOA, divided
    by the DSCR minimum), which converts to a loan via the payment factor.
    LTV sets a second, independent ceiling. The smaller wins, and which one it
    is matters as much as the number — a DSCR-bound deal gets bigger with more
    rent; an LTV-bound one does not.

    Qualifying rent is taken as gross scheduled rent, with no lender haircut —
    a simplification stated here so it isn't mistaken for a real term sheet.
    """
    qualifying_rent_annual = max(rent_m, 0.0) * 12.0
    fixed_annual = taxes + ins + hoa_m * 12.0
    allowed_annual_pi = qualifying_rent_annual / program.min_dscr - fixed_annual

    pf = payment_factor(program.rate, program.term_years, program.interest_only)
    loan_dscr = max(allowed_annual_pi, 0.0) / pf if pf > 0 else 0.0
    loan_ltv = program.max_ltv * max(price, 0.0)

    loan = max(0.0, min(loan_dscr, loan_ltv))
    binding = "dscr" if loan_dscr <= loan_ltv else "ltv"

    down_payment = max(price, 0.0) - loan
    points = loan * program.points_pct
    other_closing = max(price, 0.0) * program.other_closing_pct
    cash_to_close = down_payment + points + other_closing

    annual_pi = loan * pf
    dscr_at_loan = lender_dscr(rent_m, annual_pi, taxes, ins, hoa_m)

    return {
        "program": program.name,
        "loan": loan,
        "binding": binding,
        "down_payment": down_payment,
        "ltv": (loan / price) if price > 0 else 0.0,
        "points": points,
        "other_closing": other_closing,
        "cash_to_close": cash_to_close,
        "annual_pi": annual_pi,
        "monthly_pi": annual_pi / 12.0,
        "dscr": dscr_at_loan,
    }


def compare_programs(price: float, rent_m: float, taxes: float, ins: float,
                     programs=DEFAULT_PROGRAMS, hoa_m: float = 0.0) -> list:
    """Every program side by side, in the order given."""
    return [max_loan(price, rent_m, taxes, ins, p, hoa_m) for p in programs]


def best_program(price: float, rent_m: float, taxes: float, ins: float,
                 programs=DEFAULT_PROGRAMS, hoa_m: float = 0.0) -> dict:
    """The program that lends the most against this deal (ties broken by the
    program list order, i.e. by rate — cheapest wins a tie)."""
    rows = compare_programs(price, rent_m, taxes, ins, programs, hoa_m)
    return max(rows, key=lambda r: r["loan"])


def breakeven_rent(price: float, target_ltv: float, taxes: float, ins: float,
                   program: DscrProgram, hoa_m: float = 0.0) -> float:
    """The monthly rent needed to qualify for `target_ltv` under this program
    (capped by the program's own max_ltv — asking for more leverage than the
    program allows has no rent that answers it, so callers should compare
    `target_ltv` against `program.max_ltv` themselves)."""
    pf = payment_factor(program.rate, program.term_years, program.interest_only)
    loan_target = target_ltv * max(price, 0.0)
    allowed_annual_pi = loan_target * pf
    fixed_annual = taxes + ins + hoa_m * 12.0
    qualifying_rent_annual = program.min_dscr * (allowed_annual_pi + fixed_annual)
    return qualifying_rent_annual / 12.0


def prepay_penalty(balance: float, year: int, program: DscrProgram) -> float:
    """Real money most tools ignore: DSCR loans standardly carry a step-down
    prepayment penalty (5/4/3/2/1 by default) on the balance if refinanced or
    sold within the window. `year` is 1-based (year 1 = the step_down[0]
    rate). Zero once past the schedule."""
    step = program.prepay_step_down
    if year < 1 or year > len(step):
        return 0.0
    return balance * step[year - 1]
