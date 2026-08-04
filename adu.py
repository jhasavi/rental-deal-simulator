"""Accessory dwelling units — does adding a second rent make the deal work?

The town screener established that at today's rates no Massachusetts town cash
flows on a typical single-family with one rent. This is the most promising way
out, and it is a change in the law rather than a change in the market.

WHAT CHANGED. The Affordable Homes Act (Chapter 150 of the Acts of 2024)
amended M.G.L. c.40A §§1A and 3, effective 2 February 2025. A single accessory
dwelling unit is now allowed **by right** in every single-family zoning
district in the state — no special permit, no discretionary approval. Boston
sits outside it with its own ordinance. Regulations at 760 CMR 71.00.

The three provisions that matter to an investor:

  * **No owner-occupancy.** Municipalities may not require that the owner live
    in either unit. This is the one that makes it an investment rather than a
    homeowner's project — you can buy a single-family, add an ADU, and rent
    both.
  * **Size cap: whichever is SMALLER of 900 sq ft or half the principal
    dwelling's gross floor area.** Note "smaller" — a full 900 sq ft ADU needs
    a principal dwelling of at least 1,800 sq ft. Several published summaries
    get this backwards; the statute is explicit.
  * **Parking: none may be required within half a mile of transit**, and at
    most one space elsewhere.

Towns still control height, setbacks, lot coverage, historic districts, design
standards and building code, so nothing here is a guarantee for a given lot.
This models the economics, not the permit.

Adoption is still almost nothing — 844 applications statewide in the first six
months, against roughly a million and a half single-family homes. The
opportunity is wide open mainly because nobody has re-underwritten for it yet.

NOT LEGAL ADVICE. Confirm the local bylaw and the lot before committing money.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- the law
STATUTORY_MAX_SQFT = 900.0
PRINCIPAL_SHARE = 0.5              # ADU may be at most half the main house

# ---------------------------------------------------------------- build costs
# Massachusetts, 2026 dollars, per square foot. Ranges are wide because the
# route matters far more than the finishes: reusing an existing shell avoids
# foundation, framing and roof entirely.
BUILD_COSTS = {
    "basement or attic conversion": (150.0, 250.0),
    "attached addition": (250.0, 350.0),
    "detached new build": (275.0, 450.0),
    "garage conversion": (180.0, 300.0),
}
DEFAULT_BUILD_TYPE = "basement or attic conversion"

# Financing the build. A HELOC or construction loan is the usual route.
DEFAULT_BUILD_RATE = 0.0775
DEFAULT_BUILD_TERM_YEARS = 20

# Operating assumptions for the added unit, matching the simulator's
# conventions: maintenance is a share of scheduled rent, management a share of
# what actually gets collected.
DEFAULT_VACANCY = 0.05
DEFAULT_MAINTENANCE = 0.08
DEFAULT_MANAGEMENT = 0.08

# An ADU raises the assessment. Assuming it is assessed at what it cost to
# build is the conservative assumption.
ASSESSMENT_RATIO = 1.0
DEFAULT_TAX_RATE = 0.011           # statewide-typical residential, share of value
DEFAULT_INSURANCE_DELTA = 0.004    # added insurance, share of build cost


@dataclass
class ADUPlan:
    """One way of adding a unit to a specific house."""
    principal_sqft: float
    adu_sqft: float
    build_type: str
    build_cost: float
    financed_pct: float = 0.80
    build_rate: float = DEFAULT_BUILD_RATE
    build_term_years: int = DEFAULT_BUILD_TERM_YEARS
    tax_rate: float = DEFAULT_TAX_RATE


# ---------------------------------------------------------------- feasibility
def max_adu_sqft(principal_sqft: float) -> float:
    """The largest protected-use ADU this house may have.

    Smaller of 900 sq ft and half the principal dwelling — so anything under
    1,800 sq ft is capped by the house, not by the statute.
    """
    if principal_sqft is None or principal_sqft <= 0:
        return 0.0
    return min(STATUTORY_MAX_SQFT, principal_sqft * PRINCIPAL_SHARE)


def build_cost(adu_sqft: float, build_type: str = DEFAULT_BUILD_TYPE,
               position: float = 0.5) -> float:
    """Estimated cost to build. `position` picks a point in the range:
    0.0 the cheap end, 1.0 the dear end, 0.5 the middle."""
    if build_type not in BUILD_COSTS:
        raise ValueError(f"unknown build type: {build_type!r}")
    lo, hi = BUILD_COSTS[build_type]
    return adu_sqft * (lo + (hi - lo) * position)


def plan_for(principal_sqft: float, build_type: str = DEFAULT_BUILD_TYPE,
             position: float = 0.5, **kw) -> ADUPlan:
    """Largest legal ADU for this house, costed."""
    sqft = max_adu_sqft(principal_sqft)
    return ADUPlan(principal_sqft=principal_sqft, adu_sqft=sqft,
                   build_type=build_type,
                   build_cost=build_cost(sqft, build_type, position), **kw)


# ---------------------------------------------------------------- economics
def monthly_loan_payment(principal: float, rate: float, term_years: int) -> float:
    if principal <= 0:
        return 0.0
    if rate <= 0:
        return principal / (term_years * 12)
    r = rate / 12.0
    n = term_years * 12
    return principal * r / (1 - (1 + r) ** -n)


def carrying_cost(plan: ADUPlan) -> dict:
    """What the ADU costs you every month before any rent comes in."""
    financed = plan.build_cost * plan.financed_pct
    cash_in = plan.build_cost - financed

    loan = monthly_loan_payment(financed, plan.build_rate, plan.build_term_years)
    added_tax = plan.build_cost * ASSESSMENT_RATIO * plan.tax_rate / 12.0
    added_ins = plan.build_cost * DEFAULT_INSURANCE_DELTA / 12.0

    return {"cash_invested": cash_in, "loan_payment": loan,
            "added_tax": added_tax, "added_insurance": added_ins,
            "fixed_monthly": loan + added_tax + added_ins}


def _keep_rate(vacancy: float, management: float, maintenance: float) -> float:
    """Share of scheduled rent that survives vacancy and the two fee lines."""
    return (1 - vacancy) * (1 - management) - maintenance


def net_monthly(plan: ADUPlan, adu_rent: float,
                vacancy: float = DEFAULT_VACANCY,
                maintenance: float = DEFAULT_MAINTENANCE,
                management: float = DEFAULT_MANAGEMENT) -> float:
    """Cash the ADU puts in your pocket each month, after everything."""
    costs = carrying_cost(plan)
    kept = adu_rent * _keep_rate(vacancy, management, maintenance)
    return kept - costs["fixed_monthly"]


def breakeven_rent(plan: ADUPlan, vacancy: float = DEFAULT_VACANCY,
                   maintenance: float = DEFAULT_MAINTENANCE,
                   management: float = DEFAULT_MANAGEMENT):
    """The rent the ADU must achieve to wash its own face.

    This is the honest way to handle the biggest unknown in the model. Rather
    than guessing what a new small unit rents for in a given town and dressing
    the guess up as a projection, state the threshold and let the operator —
    who knows the street — judge whether it clears. Returns None if the fee
    structure eats every dollar of rent.
    """
    keep = _keep_rate(vacancy, management, maintenance)
    if keep <= 0:
        return None
    return carrying_cost(plan)["fixed_monthly"] / keep


def marginal_return(plan: ADUPlan, adu_rent: float,
                    vacancy: float = DEFAULT_VACANCY,
                    maintenance: float = DEFAULT_MAINTENANCE,
                    management: float = DEFAULT_MANAGEMENT) -> dict:
    """Return on the ADU capital alone.

    Deliberately separate from whatever the underlying house does. The house is
    usually either already owned or a decision made on other grounds; the live
    question is what the *next* dollar earns. That makes this comparable
    against buying another property, and it is the number that decides whether
    to build.
    """
    costs = carrying_cost(plan)
    net = net_monthly(plan, adu_rent, vacancy, maintenance, management)
    annual = net * 12.0

    cash = costs["cash_invested"]
    coc = annual / cash if cash > 0 else None

    return {
        "build_cost": plan.build_cost,
        "cash_invested": cash,
        "monthly_net": net,
        "annual_net": annual,
        "cash_on_cash": coc,
        "breakeven_rent": breakeven_rent(plan, vacancy, maintenance, management),
        "yield_on_cost": (adu_rent * 12) / plan.build_cost
        if plan.build_cost > 0 else None,
        **costs,
    }
