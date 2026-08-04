# Where to buy — the Massachusetts town screener

The simulator answers *"is this house a good buy?"*. It could never answer the
question that comes first: *"where should I be looking?"* This closes that gap.

## Running it

```bash
python3 market_data.py          # download / refresh the data (~15s, cached)
python3 town_screener.py        # ranked table in the terminal
```

Or open the **Where to buy** panel at the top of the app:

```bash
streamlit run rental_deal_simulator.py
```

Useful flags:

```bash
python3 town_screener.py --county Middlesex --top 10
python3 town_screener.py --min-yield 0.06
```

## Where the numbers come from

Everything is free, public, and needs no key, login, or MLS licence — which
means any of it can be put in front of a client without a data-rights argument.

| Source | What it gives us |
| --- | --- |
| Zillow ZHVI | Typical home value per town, monthly |
| Zillow ZORI | Typical market rent per town, monthly |
| Zillow inventory | Homes for sale |
| Zillow days-to-pending | How fast the market clears |

Coverage: 354 MA towns have home values, 141 also have rents. Rent is the
binding constraint, so 141 is the working universe, and 85 of those have enough
of the other measures to rank honestly.

## How a town is scored

Four measures, each scored against the rest of Massachusetts, then blended:

| Weight | Measure | Why |
| --- | --- | --- |
| 40% | Rent yield | A year of rent over the price. The cash-flow engine. |
| 25% | Yield direction vs 3 years ago | Rising means prices have fallen behind rents — the closest honest definition of "underpriced". |
| 20% | Rent growth, past year | Is the rent side getting stronger. |
| 15% | Buyer leverage | Homes sitting longer and inventory building = negotiating room. |

Two deliberate design decisions worth knowing about:

**Missing data is not treated as average.** A town measured on one component
would otherwise ride that single number to the top of the list. The score is a
weighted mean over whatever components exist, and `confidence` reports how much
of the signal was actually there. Towns under 60% confidence are hidden by
default (toggle in the app, `min_confidence=0.0` in code).

**The 0–100 rating is scaled after filtering**, so it always describes the
towns you are actually looking at. Filter to Middlesex and the best town in
Middlesex reads 100.

## The reality check — read this part

Ranking towns says which is *least bad*. It does not say the top one is a buy.
So the app runs the selected town's own typical price and rent through the real
underwriter and states the result plainly.

As of June 2026 data, at 25% down and 6.75%, **not one of the top-ranked towns
cash flows on a typical single-family home**:

| Town | Typical price | Break-even price | Discount needed |
| --- | --- | --- | --- |
| Springfield | $308,418 | $242,047 | −22% |
| Brockton | $516,482 | $326,062 | −37% |
| Randolph | $600,385 | $355,829 | −41% |
| Fitchburg | $409,830 | $243,619 | −41% |
| New Bedford | $449,676 | $259,595 | −42% |
| Amherst | $552,869 | $317,640 | −43% |
| Barnstable | $731,907 | $397,280 | −46% |
| Winthrop | $699,393 | $347,110 | −50% |

That table is the actual finding. At today's rates the typical Massachusetts
house does not work as a single-unit rental anywhere — so the edge is not in
picking a town, it is in **buying below typical** (distressed, estate,
long-sitting) or in **collecting more than one rent** (multi-family). The
break-even column is the offer ceiling for the first of those.

Springfield needing only 22% where everywhere else needs 37–50% is the one
genuine outlier on this list.

## Limits, stated plainly

- **Yield here is gross.** No taxes, insurance, vacancy, or management. MA
  property tax rates vary by more than 2x between towns, which is routinely the
  whole margin. The simulator handles that; the screen does not.
- **Rent is one unit.** ZORI is the typical rent for *a* rental home. A
  three-family collects roughly three of those but also costs more than the
  typical home, so the screen understates multi-family and should not be
  naively multiplied by unit count.
- **Town-wide typicals, not listings.** These pick where to look. Underwrite
  the actual listing before believing anything.
- **The break-even price assumes statewide-average tax and insurance rates**
  (1.1% and 0.5% of value). Use the real figures for a real deal.

## Tests

```bash
python3 -m pytest test_town_screener.py test_reality_check.py -q
```

28 tests covering the yield arithmetic, the missing-data rules, and the
directional properties of the break-even solver (more rent supports a higher
offer, a higher rate supports a lower one, and the returned price is genuinely
where cash flow crosses zero).
