# The DSCR loan — sizing it, not just grading it

## The finding

The simulator's own default deal — $650,000, $4,800/mo rent, 25% down,
6.75%, 30-year — reads two different ways depending on which DSCR you
compute:

| Convention | What it divides | Ratio | What it means |
| --- | --- | --- | --- |
| Investor (what the tool computed before this) | NOI (net of vacancy, management, maintenance) ÷ principal & interest | **1.02** | "Most rental-loan lenders would decline this" |
| Lender (what a DSCR underwriter actually computes) | Gross scheduled rent ÷ PITIA (principal, interest, taxes, insurance, HOA) | **1.21** | Fundable at a 1.15 or 1.00 program |

Both numbers are real. The investor number is the stricter, all-in test of
whether the property carries itself as a business. The lender number is
what's on the loan officer's worksheet. Reporting only the first one was
telling clients a financeable deal wasn't financeable — that's the bug this
closes.

## Why they disagree

Vacancy, property management and maintenance are all real costs. A DSCR
lender doesn't ask about them — they qualify the loan on the appraiser's
market-rent figure, gross, against principal, interest, taxes, insurance and
HOA. Nothing else. That's a lighter bar than the honest cash-flow question,
and the gap between the two numbers is exactly the money a real operator
spends that a DSCR lender ignores.

## From ratio to loan

A DSCR program doesn't grade a down payment you already picked — it answers
*how much would you lend*, and that number comes from two independent
constraints:

* **DSCR floor.** Whatever's left of qualifying rent after taxes,
  insurance and HOA, divided by the program's minimum DSCR, sets an
  allowance for the annual payment. That allowance, run through the
  program's rate and term, is the DSCR-constrained loan.
* **Loan-to-value cap.** A flat share of price, independent of rent.

The smaller of the two is what the program actually lends, and *which one
binds* matters as much as the number. A DSCR-bound loan gets bigger with
more rent. An LTV-bound one doesn't — more rent just leaves room under the
floor that leverage never uses.

## Interest-only is the lever, not the finish

Interest-only strips the principal component out of the required payment, so
the same DSCR floor supports a bigger loan — with no change to the rent, the
price, or the DSCR minimum. It's the one variable in this whole model that
moves a DSCR-bound loan without changing the deal underneath it. It also
means the balance never falls during the IO window, and the loan resets to
amortizing (over a shorter remaining term, so payments jump) once it ends.

## Prepayment penalties are real money

DSCR loans standardly carry a step-down prepayment penalty (5/4/3/2/1 of the
balance by year, in the default programs here) if the loan is refinanced or
sold within the window. The simulator's refinance feature now nets this out
of the cash pulled at refi — previously that cash-out showed as free money.

## The three tiers

Editable market-standard defaults, not a term sheet:

| Tier | Min DSCR | Max LTV | Rate |
| --- | --- | --- | --- |
| Best rate | 1.25 | 70% | 7.75% |
| Standard | 1.15 | 75% | 8.00% |
| Max leverage | 1.00 | 80% | 8.50% |

Looser DSCR and higher leverage cost more, in the usual way. Edit these in
the app (**Assumptions → DSCR loan → Edit program terms**) to match an
actual lender's sheet before quoting a client.

## Using it

Toggle **Finance with a DSCR loan** in the sidebar's Assumptions tab. When
on, the chosen program sizes the purchase loan itself — rate, leverage,
interest-only and points all come from the program, and the manual rate /
down payment / term fields are ignored for the purchase loan (they still
apply if the toggle is off). Section 4, **What a lender will actually
lend**, shows the max loan, which constraint binds, cash to close, and both
DSCR conventions side by side — plus how all three tiers price against the
same deal.

## Limits, stated plainly

- **No rent haircut.** Some lenders discount appraised market rent (commonly
  75%) before qualifying. This model uses gross rent as-is; a stricter
  lender will qualify a smaller loan than shown here.
- **No reserve requirement.** Most DSCR programs require 3–12 months of
  PITIA in reserves at closing. Not modeled — budget for it separately.
- **Not a rate quote.** Real DSCR pricing moves with credit score,
  prepayment structure, property type and loan size, day to day. The three
  tiers are a starting point for a conversation, not the conversation.
- **The lender-convention chart still uses year-1 numbers.** DSCR is
  underwritten once, at origination, on the appraised rent — it isn't
  re-tested every year the way the chart's later years might suggest.

## Tests

```bash
.venv/bin/python -m pytest test_dscr.py -q
```

23 tests. The headline gap (1.02 vs 1.21) is pinned first; the rest protect
the closed-form sizing math — which constraint binds, monotonicity in rate,
rent and DSCR minimum, interest-only sizing a bigger loan than amortizing,
and the prepayment step-down schedule.
