# The ADU play

## What changed in the law

The Affordable Homes Act (Chapter 150 of the Acts of 2024) amended M.G.L.
c.40A §§1A and 3, effective **2 February 2025**. Regulations at 760 CMR 71.00.

A single accessory dwelling unit is now allowed **by right** in every
single-family zoning district in Massachusetts — no special permit, no
discretionary approval. Boston sits outside it under its own ordinance.

Three provisions decide the economics:

| Rule | Detail | Why it matters |
| --- | --- | --- |
| **No owner-occupancy** | Towns may not require the owner live in either unit | This is what makes it an *investment*, not a homeowner's project. Buy a single-family, rent both units. |
| **Size cap** | Smaller of 900 sq ft **or half** the principal dwelling | Note *smaller*. A full 900 sq ft unit needs an 1,800+ sq ft house. Several published summaries state this backwards. |
| **Parking** | None may be required within ½ mile of transit; at most one space elsewhere | Removes the usual killer for infill units |

Towns still control height, setbacks, lot coverage, historic districts, design
review and building code. This models the economics, not the permit.

**Adoption is almost nothing.** 844 applications statewide in the first six
months, against roughly 1.5 million single-family homes. The opportunity is
open mainly because nobody has re-underwritten for it.

*Not legal advice. Confirm the local bylaw and the lot before committing money.*

## Why this matters here

The town screener established that at ~6.75% **no Massachusetts town cash flows
on a typical single-family with one rent**. An ADU is the most promising way
out, because it is a change in the law rather than a change in the market.

## The finding: ADU economics run opposite to purchase economics

Build cost barely varies across the state. Rents vary enormously. So the towns
worth *buying* cheaply are the worst places to *add* a unit, and the expensive
suburbs are the best.

Break-even rent for a 900 sq ft basement/attic conversion at $180k, 80%
financed at 7.75%, is **$1,772/mo**. Against typical town rents:

| Town | Typical rent | Break-even as % of it | Verdict |
| --- | --- | --- | --- |
| Weston | $6,900 | 26% | huge cushion |
| Lexington | $4,274 | 41% | huge cushion |
| Wellesley | $3,868 | 46% | huge cushion |
| Newton | $3,685 | 48% | strong |
| Brookline | $3,523 | 50% | strong |
| Needham | $3,155 | 56% | strong |
| Brockton | $2,406 | 74% | workable, thin |
| Taunton | $2,124 | 83% | thin |
| New Bedford | $1,916 | 93% | no margin |
| Springfield | $1,786 | 99% | dead |

The robust claim does not depend on guessing what a new unit rents for. In
Lexington the unit must clear 41% of the town's typical rent — it could let at
well under market and still carry itself. In Springfield it must clear 99%,
meaning a *smaller* unit has to match a typical whole home. That never happens.

## The strategic point

The best ADU towns — Newton, Needham, Wellesley, Lexington, Brookline — are
exactly the Greater Boston suburbs where the brokerage already has a community
and a client base. And the multi-generational angle (space for visiting or
immigrating parents) means the same build serves a family need *and* an
investment return, which is a far easier conversation than either alone.

## The build route matters more than the finishes

Reusing an existing shell skips foundation, framing and roof:

| Route | $/sq ft | 900 sq ft | Break-even rent |
| --- | --- | --- | --- |
| Basement or attic conversion | $150–250 | $180,000 | $1,772 |
| Garage conversion | $180–300 | $216,000 | $2,127 |
| Attached addition | $250–350 | $270,000 | $2,658 |
| Detached new build | $275–450 | $326,250 | $3,212 |

At a detached new build's $3,212 break-even, only a handful of towns in the
state clear it. **Conversions work; new builds mostly don't.**

## Leverage helps but is not the story

900 sq ft conversion, $180k, at $2,800/mo rent:

| Financed | Your cash | Break-even rent | Net/mo | Cash-on-cash |
| --- | --- | --- | --- | --- |
| 0% | $180,000 | $283 | $1,998 | 13.3% |
| 50% | $90,000 | $1,214 | $1,259 | 16.8% |
| 80% | $36,000 | $1,772 | $816 | 27.2% |
| 100% | $0 | $2,144 | $520 | n/a — no cash in |

All-cash yield on cost is **18.7%**. An unlevered unit only has to cover its
own tax and insurance, hence the $283 break-even. Borrowing raises the return
and the risk together, in the usual way.

## Using it

The **Add a second unit** panel inside *Where to buy*. Set the house size, the
build route, and how much you'd borrow; it returns the largest legal unit, the
cost, and the rent it must clear — plus a statewide ranking by cushion.

One design note worth knowing: that ranking covers all 141 towns with a rent
figure, **not** the 85 that pass the buying table's confidence gate. That gate
measures coverage of the four purchase signals, which say nothing about whether
a unit would let. Applying it here hid Lexington and Weston — the two best
candidates in the state.

## Limits

- Town-wide typical rents, not comps for a specific unit. Get real comps.
- Assumes the ADU is assessed at build cost for the tax increase.
- Says nothing about whether a given lot passes setbacks, height or historic
  review — the parts towns still control.
- Ignores the build period: 4–10 months of carrying cost with no rent.

## Tests

```bash
python3 -m pytest test_adu.py -q
```

17 tests, with the statutory size cap pinned hardest — including that a
1,200 sq ft house gets 600 sq ft and not 900.
