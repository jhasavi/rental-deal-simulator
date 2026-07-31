"""
Rental Deal Simulator — Monte Carlo underwriting for rental / investor deals.
Namaste Boston Homes.

Run locally:
    streamlit run rental_deal_simulator.py

Single-file Streamlit app. The simulation core is pure NumPy (vectorized across
trials) and importable without Streamlit for testing:

    from rental_deal_simulator import DealInputs, Assumptions, run_simulation

v2 features: stochastic appreciation, after-tax returns, two-deal compare,
DSCR lender view, mid-hold refinance, and branded PDF export.

NOT TAX ADVICE. The tax layer is a simplified model for screening deals.
Every real deal should be reviewed by a CPA.
"""

import io
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
RENT_GROWTH_STDEV = 0.015          # year-to-year st.dev around the growth assumption
REPAIR_PROB_RANGE = (0.02, 0.04)   # annual chance of a big repair (drawn per trial)
REPAIR_COST_RANGE = (5_000, 20_000)
EXPENSE_INFLATION = 0.025          # taxes / insurance / HOA annual inflation
DEPRECIATION_YEARS = 27.5          # residential straight-line

# chart palette (dataviz reference palette: slot 1 blue, slot 2 orange, chrome)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
LIGHT_BLUE = "#86b6ef"
MUTED = "#898781"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


# ---------------------------------------------------------------- inputs
@dataclass
class DealInputs:
    """Per-property inputs. Two of these are compared side by side."""
    label: str = "Deal A"
    address: str = ""
    purchase_price: float = 650_000.0
    down_payment_pct: float = 0.25      # fraction of price
    rate: float = 0.0675                # annual mortgage rate
    term_years: int = 30
    monthly_rent: float = 4_800.0
    taxes_annual: float = 7_200.0
    insurance_annual: float = 2_400.0
    hoa_monthly: float = 0.0
    vacancy_pct: float = 0.05           # probability any month is vacant
    maintenance_pct: float = 0.08       # % of scheduled rent
    management_pct: float = 0.08        # % of collected rent
    rent_growth: float = 0.03           # annual assumption (mean)


@dataclass
class Assumptions:
    """Market / investor assumptions, shared across both deals so an A-vs-B
    comparison stays apples-to-apples."""
    years: int = 10
    n_trials: int = 5_000
    seed: int = 42

    # appreciation & exit
    appreciation_mean: float = 0.04     # Greater Boston long-run nominal
    appreciation_stdev: float = 0.07    # year-to-year volatility
    selling_cost_pct: float = 0.05      # commission + transfer + legal at exit

    # tax layer
    tax_enabled: bool = False
    marginal_tax_rate: float = 0.32     # combined fed + MA on ordinary income
    land_pct: float = 0.20              # non-depreciable land share of price
    capital_gains_rate: float = 0.20    # long-term cap gains at sale
    recapture_rate: float = 0.25        # depreciation recapture at sale

    # refinance
    refi_enabled: bool = False
    refi_year: int = 3
    refi_rate: float = 0.055
    refi_term_years: int = 30
    refi_ltv: float = 0.75
    refi_cost_pct: float = 0.02         # closing costs as % of new loan

    # DSCR lender view
    dscr_thresholds: tuple = (1.15, 1.25)


# ---------------------------------------------------------------- core math
def monthly_payment(loan, rate: float, term_years: int):
    """Level P&I payment. `loan` may be a scalar or an array."""
    loan = np.asarray(loan, dtype=float)
    n = term_years * 12
    r = rate / 12
    if r == 0:
        pmt = loan / n
    else:
        pmt = loan * r / (1 - (1 + r) ** -n)
    return np.where(loan > 0, pmt, 0.0)


def amortize(loan, rate: float, term_years: int, n_months: int):
    """Vectorized amortization.

    `loan` is a scalar or (n_trials,) array. Returns:
      payment  (n, n_months) — 0 once the loan is retired
      interest (n, n_months)
      balance  (n,)          — remaining balance after n_months payments
    """
    loan = np.atleast_1d(np.asarray(loan, dtype=float)).reshape(-1, 1)
    r = rate / 12
    n_term = term_years * 12
    pmt = monthly_payment(loan, rate, term_years)          # (n, 1)
    m = np.arange(n_months)[None, :]                       # (1, n_months)

    if r == 0:
        bal = loan - pmt * m
        interest = np.zeros((loan.shape[0], n_months))
        bal_end = loan[:, 0] - pmt[:, 0] * min(n_months, n_term)
    else:
        grow = (1 + r) ** m
        bal = loan * grow - pmt * (grow - 1) / r           # balance before pmt m+1
        interest = np.maximum(bal, 0.0) * r
        g_end = (1 + r) ** min(n_months, n_term)
        bal_end = loan[:, 0] * g_end - pmt[:, 0] * (g_end - 1) / r

    active = (m < n_term) & (bal > 1e-6)
    payment = np.where(active, pmt, 0.0)
    interest = np.where(active, interest, 0.0)
    return payment, interest, np.maximum(bal_end, 0.0)


def _vectorized_irr(flows: np.ndarray, lo: float = -0.99, hi: float = 5.0,
                    iters: int = 100) -> np.ndarray:
    """IRR per row of `flows` (col 0 = t0) via bisection; NaN where unbracketed."""
    t = np.arange(flows.shape[1])

    def npv(rates):
        return (flows / (1.0 + rates[:, None]) ** t).sum(axis=1)

    lo_a = np.full(flows.shape[0], lo)
    hi_a = np.full(flows.shape[0], hi)
    f_lo, f_hi = npv(lo_a), npv(hi_a)
    ok = (np.sign(f_lo) != np.sign(f_hi)) & np.isfinite(f_lo) & np.isfinite(f_hi)
    for _ in range(iters):
        mid = (lo_a + hi_a) / 2
        f_mid = npv(mid)
        go_lo = np.sign(f_mid) == np.sign(f_lo)
        lo_a = np.where(go_lo, mid, lo_a)
        f_lo = np.where(go_lo, f_mid, f_lo)
        hi_a = np.where(go_lo, hi_a, mid)
    irr = (lo_a + hi_a) / 2
    irr[~ok] = np.nan
    return irr


def _depreciation_by_year(building_basis: float, years: int) -> np.ndarray:
    """Straight-line over 27.5 years, pro-rated in the final partial year."""
    per_year = building_basis / DEPRECIATION_YEARS
    edges = np.minimum(np.arange(0, years + 1, dtype=float), DEPRECIATION_YEARS)
    return per_year * np.diff(edges)


def run_simulation(inp: DealInputs, a: Assumptions = None) -> dict:
    """Vectorized Monte Carlo over `a.n_trials` trials. Returns arrays by metric."""
    a = a or Assumptions()
    rng = np.random.default_rng(a.seed)
    years, n = int(a.years), int(a.n_trials)
    months = years * 12

    cash_invested = inp.purchase_price * inp.down_payment_pct
    loan0 = inp.purchase_price - cash_invested

    # ---- stochastic property value (drawn first so the same seed gives the
    # same appreciation path across deals — a paired A/B comparison)
    appr = rng.normal(a.appreciation_mean, a.appreciation_stdev, size=(n, years))
    value_by_year = inp.purchase_price * np.cumprod(1.0 + appr, axis=1)   # (n, years)

    # ---- rent path
    growth = rng.normal(inp.rent_growth, RENT_GROWTH_STDEV, size=(n, years - 1))
    mult = np.concatenate([np.ones((n, 1)), np.cumprod(1.0 + growth, axis=1)], axis=1)
    sched_rent_m = np.repeat(inp.monthly_rent * mult, 12, axis=1)         # (n, months)

    # ---- vacancy: independent Bernoulli per month
    occupied = rng.random((n, months)) >= inp.vacancy_pct
    collected = sched_rent_m * occupied

    # ---- operating expenses
    mgmt = inp.management_pct * collected
    maint = inp.maintenance_pct * sched_rent_m
    year_idx = np.repeat(np.arange(years), 12)
    fixed_m = ((inp.taxes_annual + inp.insurance_annual) / 12 + inp.hoa_monthly) \
        * (1 + EXPENSE_INFLATION) ** year_idx                            # (months,)

    # NOI excludes debt service and big repairs (lender convention for DSCR)
    noi_m = collected - mgmt - maint - fixed_m
    noi = noi_m.reshape(n, years, 12).sum(axis=2)                        # (n, years)

    # ---- debt: original loan, optionally refinanced at year N
    payment, interest, _ = amortize(np.full(n, loan0), inp.rate,
                                    inp.term_years, months)
    refi_cash = np.zeros((n, years))
    refi_month = None

    if a.refi_enabled and 0 < a.refi_year < years and loan0 > 0:
        refi_month = int(a.refi_year) * 12
        # balance at the refi date under the original loan
        _, _, bal_at_refi = amortize(np.full(n, loan0), inp.rate,
                                     inp.term_years, refi_month)
        value_at_refi = value_by_year[:, int(a.refi_year) - 1]
        new_loan = a.refi_ltv * value_at_refi
        costs = a.refi_cost_pct * new_loan
        # cash pulled out (negative = cash the investor must bring to close)
        refi_cash[:, int(a.refi_year) - 1] = new_loan - bal_at_refi - costs

        pay2, int2, _ = amortize(new_loan, a.refi_rate, a.refi_term_years,
                                 months - refi_month)
        payment = np.concatenate([payment[:, :refi_month], pay2], axis=1)
        interest = np.concatenate([interest[:, :refi_month], int2], axis=1)
        _, _, bal_end = amortize(new_loan, a.refi_rate, a.refi_term_years,
                                 months - refi_month)
    else:
        _, _, bal_end = amortize(np.full(n, loan0), inp.rate,
                                 inp.term_years, months)

    debt_service = payment.reshape(n, years, 12).sum(axis=2)             # (n, years)
    interest_yr = interest.reshape(n, years, 12).sum(axis=2)

    # ---- big repair events
    p_repair = rng.uniform(*REPAIR_PROB_RANGE, size=(n, 1))
    repair_hit = rng.random((n, years)) < p_repair
    repair_cost = repair_hit * rng.uniform(*REPAIR_COST_RANGE, size=(n, years))

    # ---- operating cash flow (refi proceeds tracked separately so they don't
    # distort cash-on-cash; they enter the IRR flows below)
    annual_cf = noi - debt_service - repair_cost                         # (n, years)

    # ---- DSCR (lender view): NOI over debt service
    with np.errstate(divide="ignore", invalid="ignore"):
        dscr = np.where(debt_service > 0, noi / debt_service, np.inf)

    # ---- exit
    gross_sale = value_by_year[:, -1]
    net_sale = gross_sale * (1 - a.selling_cost_pct)
    equity_at_exit = net_sale - bal_end

    # ---- tax layer
    tax_by_year = np.zeros((n, years))
    sale_tax = np.zeros(n)
    dep_by_year = _depreciation_by_year(
        inp.purchase_price * (1 - a.land_pct), years)
    if a.tax_enabled:
        deductible_opex = (mgmt + maint).reshape(n, years, 12).sum(axis=2) \
            + fixed_m.reshape(years, 12).sum(axis=1)[None, :] + repair_cost
        rental_income = collected.reshape(n, years, 12).sum(axis=2)
        taxable = rental_income - deductible_opex - interest_yr - dep_by_year[None, :]

        # passive losses suspend and carry forward until income or sale
        carry = np.zeros(n)
        for y in range(years):
            inc = taxable[:, y]
            pos = np.maximum(inc, 0.0)
            used = np.minimum(carry, pos)
            carry = carry - used + np.maximum(-inc, 0.0)
            tax_by_year[:, y] = (pos - used) * a.marginal_tax_rate

        accum_dep = dep_by_year.sum()
        adjusted_basis = inp.purchase_price - accum_dep
        gain = net_sale - adjusted_basis
        recapture = np.clip(np.minimum(accum_dep, gain), 0.0, None)
        cap_gain = np.clip(gain - recapture, 0.0, None)
        loss_on_sale = np.clip(-gain, 0.0, None)
        sale_tax = (recapture * a.recapture_rate
                    + cap_gain * a.capital_gains_rate
                    - loss_on_sale * a.marginal_tax_rate
                    - carry * a.marginal_tax_rate)   # suspended losses release

    after_tax_cf = annual_cf - tax_by_year
    equity_after_tax = equity_at_exit - sale_tax

    # ---- returns
    coc = annual_cf / cash_invested
    avg_coc = annual_cf.mean(axis=1) / cash_invested
    avg_coc_at = after_tax_cf.mean(axis=1) / cash_invested
    p_positive = (annual_cf > 0).mean(axis=0)

    def _irr(cf_stream, exit_equity):
        flows = np.concatenate(
            [np.full((n, 1), -cash_invested),
             (cf_stream + refi_cash)[:, :-1],
             (cf_stream[:, -1] + refi_cash[:, -1] + exit_equity)[:, None]], axis=1)
        return _vectorized_irr(flows)

    return {
        "inputs": inp,
        "assumptions": a,
        "annual_cf": annual_cf,
        "after_tax_cf": after_tax_cf,
        "refi_cash": refi_cash,
        "coc": coc,
        "avg_coc": avg_coc,
        "avg_coc_after_tax": avg_coc_at,
        "irr": _irr(annual_cf, equity_at_exit),
        "irr_after_tax": _irr(after_tax_cf, equity_after_tax),
        "p_positive": p_positive,
        "dscr": dscr,
        "noi": noi,
        "debt_service": debt_service,
        "value_by_year": value_by_year,
        "equity_at_exit": equity_at_exit,
        "equity_after_tax": equity_after_tax,
        "tax_by_year": tax_by_year,
        "sale_tax": sale_tax,
        "cash_invested": cash_invested,
        "monthly_payment": float(monthly_payment(loan0, inp.rate, inp.term_years)),
        "loan0": loan0,
        "refi_month": refi_month,
        "depreciation_annual": float(dep_by_year[0]) if len(dep_by_year) else 0.0,
    }


def bad_luck_year1(inp: DealInputs, vacant_months: int = 3,
                   repair_cost: float = 12_500.0) -> dict:
    """Deterministic stress: `vacant_months` empty + one big repair in year 1,
    vs. the expected year 1 (vacancy % applied evenly, no repair)."""
    cash = inp.purchase_price * inp.down_payment_pct
    loan = inp.purchase_price - cash
    pmt = float(monthly_payment(loan, inp.rate, inp.term_years))
    fixed = inp.taxes_annual + inp.insurance_annual + 12 * inp.hoa_monthly
    sched = 12 * inp.monthly_rent

    def year_cf(collected, repair):
        return (collected - inp.management_pct * collected
                - inp.maintenance_pct * sched - fixed - 12 * pmt - repair)

    expected = year_cf(sched * (1 - inp.vacancy_pct), 0.0)
    stressed = year_cf(inp.monthly_rent * (12 - vacant_months), repair_cost)
    return {"expected": expected, "stressed": stressed,
            "delta": stressed - expected}


# ---------------------------------------------------------------- formatting
def pct(x: float, dp: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{dp}f}%"


def usd(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def ratio(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2f}x"


def md(s: str) -> str:
    """Escape literal $ before passing text to st.caption/st.markdown — two
    unescaped $ in one string (e.g. two usd() values) get parsed as a LaTeX
    math span and silently mangle the text."""
    return s.replace("$", "\\$")


def summary_row(res: dict) -> dict:
    """Headline numbers for one deal — used by the UI and the PDF."""
    a = res["assumptions"]
    irr = res["irr_after_tax"] if a.tax_enabled else res["irr"]
    coc = res["avg_coc_after_tax"] if a.tax_enabled else res["avg_coc"]
    d1 = res["dscr"][:, 0]
    return {
        "coc_mean": float(np.mean(coc)),
        "coc_p10": float(np.percentile(coc, 10)),
        "coc_p90": float(np.percentile(coc, 90)),
        "irr_mean": float(np.nanmean(irr)),
        "irr_p10": float(np.nanpercentile(irr, 10)),
        "irr_p90": float(np.nanpercentile(irr, 90)),
        "p_pos_y1": float(res["p_positive"][0]),
        "p_pos_y5": float(res["p_positive"][min(4, a.years - 1)]),
        "dscr_median": float(np.median(d1[np.isfinite(d1)])) if np.isfinite(d1).any() else np.inf,
        "dscr_pass_115": float(np.mean(d1 >= 1.15)),
        "dscr_pass_125": float(np.mean(d1 >= 1.25)),
        "cash_invested": res["cash_invested"],
        "monthly_payment": res["monthly_payment"],
        "worst_year_p10": float(np.percentile(res["annual_cf"].min(axis=1), 10)),
        "equity_mean": float(np.mean(
            res["equity_after_tax"] if a.tax_enabled else res["equity_at_exit"])),
    }


# ---------------------------------------------------------------- PDF export
def build_pdf(results: list, a: Assumptions) -> bytes:
    """One-page branded client summary. `results` is a list of run_simulation dicts."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    buf = io.BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=letter)
    W, H = letter
    brand = colors.HexColor(BLUE)
    ink = colors.HexColor("#0b0b0b")
    soft = colors.HexColor("#52514e")

    # header band
    c.setFillColor(brand)
    c.rect(0, H - 0.9 * inch, W, 0.9 * inch, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(0.6 * inch, H - 0.55 * inch, "Rental Deal Analysis")
    c.setFont("Helvetica", 9.5)
    c.drawRightString(W - 0.6 * inch, H - 0.45 * inch, "Namaste Boston Homes")
    c.drawRightString(W - 0.6 * inch, H - 0.65 * inch,
                      date.today().strftime("%B %d, %Y"))

    y = H - 1.25 * inch
    c.setFillColor(ink)
    c.setFont("Helvetica-Bold", 11)
    titles = " vs. ".join(
        (r["inputs"].address or r["inputs"].label) for r in results)
    c.drawString(0.6 * inch, y, titles[:95])
    y -= 0.22 * inch
    c.setFillColor(soft)
    c.setFont("Helvetica", 8.5)
    c.drawString(0.6 * inch, y,
                 f"{a.n_trials:,} Monte Carlo trials · {a.years}-year hold · "
                 f"appreciation {pct(a.appreciation_mean)} ± {pct(a.appreciation_stdev)}"
                 + (" · after-tax" if a.tax_enabled else " · pre-tax")
                 + (f" · refi year {a.refi_year} @ {pct(a.refi_rate, 2)}"
                    if a.refi_enabled else ""))
    y -= 0.28 * inch

    # metrics table
    rows = [["", *[(r["inputs"].label) for r in results]]]
    sums = [summary_row(r) for r in results]
    tax_tag = " (after tax)" if a.tax_enabled else ""

    def add(label, fn):
        rows.append([label, *[fn(s) for s in sums]])

    rows.append(["Purchase price",
                 *[usd(r["inputs"].purchase_price) for r in results]])
    add("Cash invested", lambda s: usd(s["cash_invested"]))
    add("Monthly P&I", lambda s: usd(s["monthly_payment"]))
    add(f"Avg cash-on-cash{tax_tag}", lambda s: pct(s["coc_mean"]))
    add("  10th–90th percentile", lambda s: f"{pct(s['coc_p10'])} to {pct(s['coc_p90'])}")
    add(f"{a.years}-yr IRR{tax_tag}", lambda s: pct(s["irr_mean"]))
    add("  10th–90th percentile", lambda s: f"{pct(s['irr_p10'])} to {pct(s['irr_p90'])}")
    add("P(positive cash flow) yr 1", lambda s: pct(s["p_pos_y1"]))
    add("P(positive cash flow) yr 5", lambda s: pct(s["p_pos_y5"]))
    add("Median DSCR (year 1)", lambda s: ratio(s["dscr_median"]))
    add("DSCR pass rate @ 1.25", lambda s: pct(s["dscr_pass_125"]))
    add("Worst year (10th pct)", lambda s: usd(s["worst_year_p10"]))
    add("Projected equity at exit", lambda s: usd(s["equity_mean"]))

    col_w = [2.5 * inch] + [2.0 * inch] * len(results)
    tbl = Table(rows, colWidths=col_w[:1 + len(results)], rowHeights=0.235 * inch)
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("FONT", (0, 1), (0, -1), "Helvetica", 9),
        ("FONT", (1, 1), (-1, -1), "Helvetica-Bold", 9.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), brand),
        ("TEXTCOLOR", (0, 1), (0, -1), soft),
        ("TEXTCOLOR", (1, 1), (-1, -1), ink),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, brand),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#e1e0d9")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    tw, th = tbl.wrapOn(c, W, H)
    tbl.drawOn(c, 0.6 * inch, y - th)
    y = y - th - 0.35 * inch

    # P(positive cash flow) by year — simple bar chart
    chart_h = 1.15 * inch
    chart_w = W - 1.2 * inch
    x0, y0 = 0.6 * inch, y - chart_h
    c.setFillColor(ink)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x0, y + 0.06 * inch, "Probability of positive cash flow by year")
    c.setStrokeColor(colors.HexColor("#e1e0d9"))
    c.setLineWidth(0.5)
    for frac in (0, 0.5, 1.0):
        c.line(x0, y0 + frac * chart_h, x0 + chart_w, y0 + frac * chart_h)
    c.setFont("Helvetica", 6.5)
    c.setFillColor(soft)
    for frac in (0, 0.5, 1.0):
        c.drawRightString(x0 - 3, y0 + frac * chart_h - 2, f"{int(frac * 100)}%")

    n_dl = len(results)
    slot = chart_w / a.years
    bw = min(0.16 * inch, (slot - 4) / n_dl)
    palette = [colors.HexColor(BLUE), colors.HexColor(ORANGE)]
    for di, r in enumerate(results):
        c.setFillColor(palette[di % 2])
        for yr in range(a.years):
            h = float(r["p_positive"][yr]) * chart_h
            bx = x0 + yr * slot + (slot - bw * n_dl) / 2 + di * bw
            c.rect(bx, y0, bw, max(h, 0.4), fill=1, stroke=0)
    c.setFillColor(soft)
    c.setFont("Helvetica", 6.5)
    for yr in range(a.years):
        c.drawCentredString(x0 + yr * slot + slot / 2, y0 - 9, str(yr + 1))

    if n_dl > 1:
        lx = x0
        for di, r in enumerate(results):
            c.setFillColor(palette[di % 2])
            c.rect(lx, y0 - 22, 7, 7, fill=1, stroke=0)
            c.setFillColor(soft)
            c.setFont("Helvetica", 7)
            c.drawString(lx + 10, y0 - 21, r["inputs"].label)
            lx += 1.3 * inch

    # assumptions + disclaimer — anchored just below the chart (and its
    # legend, when present) rather than a fixed page offset, so there's no
    # dead space on a one-deal run and no crowding on a two-deal run
    fy = y0 - (0.42 * inch if n_dl > 1 else 0.28 * inch)
    c.setFillColor(soft)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(0.6 * inch, fy, "Key assumptions")
    c.setFont("Helvetica", 7)
    d = results[0]["inputs"]
    lines = [
        f"Vacancy {pct(d.vacancy_pct)} · maintenance {pct(d.maintenance_pct)} of rent · "
        f"management {pct(d.management_pct)} of collected rent · rent growth "
        f"{pct(d.rent_growth)}/yr ± {pct(RENT_GROWTH_STDEV)}",
        f"Big repair {pct(REPAIR_PROB_RANGE[0])}–{pct(REPAIR_PROB_RANGE[1])} annual chance at "
        f"{usd(REPAIR_COST_RANGE[0])}–{usd(REPAIR_COST_RANGE[1])} · expenses inflate "
        f"{pct(EXPENSE_INFLATION)}/yr · selling costs {pct(a.selling_cost_pct)} at exit",
    ]
    if a.tax_enabled:
        lines.append(
            f"Tax: {pct(a.marginal_tax_rate)} ordinary · {pct(a.capital_gains_rate)} cap gains · "
            f"{pct(a.recapture_rate)} recapture · land {pct(a.land_pct)} of basis · "
            f"straight-line over {DEPRECIATION_YEARS} yrs")
    for i, ln in enumerate(lines):
        c.drawString(0.6 * inch, fy - 0.13 * inch * (i + 1), ln[:150])

    c.setStrokeColor(colors.HexColor("#e1e0d9"))
    c.line(0.6 * inch, 0.62 * inch, W - 0.6 * inch, 0.62 * inch)
    c.setFont("Helvetica-Oblique", 6.5)
    c.setFillColor(soft)
    disclaimer = ("Projections are model output based on the assumptions above, not a "
                  "guarantee of performance. Not tax, legal, or investment advice — "
                  "review any deal with your CPA and attorney. Namaste Boston Homes / "
                  "Namaste Boston LLC.")
    st_ = ParagraphStyle("d", fontName="Helvetica-Oblique", fontSize=6.5,
                         textColor=soft, leading=8)
    p = Paragraph(disclaimer, st_)
    pw, ph = p.wrapOn(c, W - 1.2 * inch, 0.5 * inch)
    p.drawOn(c, 0.6 * inch, 0.62 * inch - ph - 4)

    c.showPage()
    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------- charts
def histogram_df(values: np.ndarray, label: str, bins=None) -> pd.DataFrame:
    counts, edges = np.histogram(values, bins=bins if bins is not None else 40)
    gap = 0.06 * (edges[1] - edges[0])
    mid = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"lo": edges[:-1] + gap, "hi": edges[1:] - gap,
                         "count": counts, "zero": 0, "mid": mid, "Deal": label})


def dist_chart(series: list, title: str, alt, fmt: str = ".1f"):
    """Overlaid distribution for one or two deals.

    Single-layer by construction: layered Altair charts lose their data on
    Streamlit reruns in this version, so percentile context goes in the title.
    """
    all_vals = np.concatenate([v for _, v in series])
    edges = np.histogram_bin_edges(all_vals, bins=40)
    df = pd.concat([histogram_df(v, lab, bins=edges) for lab, v in series],
                   ignore_index=True)
    labels = [lab for lab, _ in series]
    caption = "   ".join(
        f"{lab}: p10 {np.percentile(v, 10):{fmt}} · median "
        f"{np.percentile(v, 50):{fmt}} · p90 {np.percentile(v, 90):{fmt}}"
        for lab, v in series)
    enc = dict(
        x=alt.X("lo:Q", title=title),
        x2=alt.X2("hi:Q"),
        y=alt.Y("count:Q", title="Trials", stack=None),
        y2=alt.Y2("zero:Q"),
        tooltip=[alt.Tooltip("Deal:N"), alt.Tooltip("mid:Q", title=title, format=fmt),
                 alt.Tooltip("count:Q", title="Trials")],
    )
    if len(series) > 1:
        enc["color"] = alt.Color(
            "Deal:N", scale=alt.Scale(domain=labels, range=[BLUE, ORANGE]),
            legend=alt.Legend(title=None, orient="top"))
        enc["opacity"] = alt.value(0.62)
    else:
        enc["color"] = alt.value(BLUE)
    return alt.Chart(df).mark_bar(
        cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(**enc).properties(
        height=260,
        title=alt.TitleParams(caption, fontSize=11, color=MUTED, anchor="start",
                              fontWeight="normal"))


def by_year_chart(frames: list, value_col: str, title: str, y_title: str, alt,
                  y_fmt: str = "$,.0f"):
    """One line per deal-and-band over the hold period. Single-layer.

    Median lines take the deal's base color; the 10th/90th percentile lines take
    a lighter step of the same hue and a dashed stroke.
    """
    df = pd.concat(frames, ignore_index=True)
    deal_order = list(dict.fromkeys(df["Deal"]))
    series_order = list(dict.fromkeys(df["Series"]))
    shades = {deal_order[0]: (BLUE, LIGHT_BLUE)}
    if len(deal_order) > 1:
        shades[deal_order[1]] = (ORANGE, "#f0a184")
    colors = []
    for s in series_order:
        row = df[df["Series"] == s].iloc[0]
        base, light = shades[row["Deal"]]
        colors.append(base if row["Band"] == "Median" else light)
    return (
        alt.Chart(df).mark_line(strokeWidth=2, point=True).encode(
            x=alt.X("Year:O", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{value_col}:Q", title=y_title),
            color=alt.Color("Series:N",
                            scale=alt.Scale(domain=series_order, range=colors),
                            legend=alt.Legend(title=None, orient="top")),
            strokeDash=alt.StrokeDash(
                "Band:N", scale=alt.Scale(domain=["Median", "Range"],
                                          range=[[1, 0], [4, 3]]), legend=None),
            tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("Series:N"),
                     alt.Tooltip(f"{value_col}:Q", title=y_title, format=y_fmt)],
        ).properties(height=280, title=title)
    )


# ---------------------------------------------------------------- UI
DEFAULT_B = DealInputs(label="Deal B", purchase_price=730_000.0,
                       monthly_rent=5_200.0, taxes_annual=8_000.0)


def deal_form(st, key: str, d: DealInputs) -> DealInputs:
    """Render inputs for one deal and return the populated DealInputs."""
    address = st.text_input("Property / address", d.address, key=f"{key}_addr",
                            placeholder="30-32 Dewey Rd, Shrewsbury")
    price = st.number_input("Purchase price ($)", 50_000, 20_000_000,
                            int(d.purchase_price), step=10_000, key=f"{key}_price")
    down = st.number_input("Down payment (%)", 0.0, 100.0,
                           d.down_payment_pct * 100, step=5.0, key=f"{key}_down")
    rate = st.number_input("Mortgage rate (%)", 0.0, 15.0, d.rate * 100,
                           step=0.125, key=f"{key}_rate")
    term = st.number_input("Loan term (years)", 5, 40, d.term_years, step=5,
                           key=f"{key}_term")
    rent = st.number_input("Monthly rent — all units ($)", 0, 100_000,
                           int(d.monthly_rent), step=50, key=f"{key}_rent")
    taxes = st.number_input("Property taxes ($/yr)", 0, 200_000,
                            int(d.taxes_annual), step=100, key=f"{key}_tax")
    ins = st.number_input("Insurance ($/yr)", 0, 100_000, int(d.insurance_annual),
                          step=100, key=f"{key}_ins")
    hoa = st.number_input("HOA ($/mo)", 0, 10_000, int(d.hoa_monthly), step=25,
                          key=f"{key}_hoa")
    vac = st.number_input("Vacancy (%)", 0.0, 50.0, d.vacancy_pct * 100, step=1.0,
                          key=f"{key}_vac")
    maint = st.number_input("Maintenance (% of rent)", 0.0, 50.0,
                            d.maintenance_pct * 100, step=1.0, key=f"{key}_maint")
    mgmt = st.number_input("Property management (% collected)", 0.0, 50.0,
                           d.management_pct * 100, step=1.0, key=f"{key}_mgmt")
    growth = st.number_input("Annual rent growth (%)", -5.0, 15.0,
                             d.rent_growth * 100, step=0.5, key=f"{key}_growth")
    return DealInputs(
        label=d.label, address=address, purchase_price=float(price),
        down_payment_pct=down / 100, rate=rate / 100, term_years=int(term),
        monthly_rent=float(rent), taxes_annual=float(taxes),
        insurance_annual=float(ins), hoa_monthly=float(hoa),
        vacancy_pct=vac / 100, maintenance_pct=maint / 100,
        management_pct=mgmt / 100, rent_growth=growth / 100)


def main():
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="Rental Deal Simulator", page_icon="🏠",
                       layout="wide")
    st.title("Rental Deal Simulator")
    st.caption("Monte Carlo underwriting — distributions, not single numbers. "
               "Namaste Boston Homes.")

    with st.sidebar:
        compare = st.toggle("Compare two deals", value=False)
        tabs = st.tabs(["Deal A", "Deal B", "Assumptions"] if compare
                       else ["Deal A", "Assumptions"])
        with tabs[0]:
            deal_a = deal_form(st, "a", DealInputs())
        deal_b = None
        if compare:
            with tabs[1]:
                deal_b = deal_form(st, "b", DEFAULT_B)
        with tabs[-1]:
            st.caption("Shared across both deals so the comparison is fair.")
            years = st.number_input("Hold period (years)", 2, 30, 10, step=1)
            n_trials = st.select_slider("Trials", [1_000, 2_500, 5_000, 10_000],
                                        value=5_000)
            st.markdown("**Appreciation & exit**")
            app_mean = st.number_input("Appreciation mean (%/yr)", -5.0, 15.0, 4.0,
                                       step=0.5)
            app_std = st.number_input("Appreciation volatility (± %/yr)", 0.0, 25.0,
                                      7.0, step=0.5)
            sell_cost = st.number_input("Selling costs at exit (%)", 0.0, 15.0, 5.0,
                                        step=0.5)
            st.markdown("**Tax layer**")
            tax_on = st.toggle("Model after-tax returns", value=False)
            marg = st.number_input("Marginal tax rate — fed + MA (%)", 0.0, 60.0,
                                   32.0, step=1.0, disabled=not tax_on)
            land = st.number_input("Land share of price (%)", 0.0, 60.0, 20.0,
                                   step=5.0, disabled=not tax_on)
            cg = st.number_input("Capital gains rate (%)", 0.0, 40.0, 20.0,
                                 step=1.0, disabled=not tax_on)
            recap = st.number_input("Depreciation recapture rate (%)", 0.0, 40.0,
                                    25.0, step=1.0, disabled=not tax_on)
            st.markdown("**Refinance**")
            refi_on = st.toggle("Refinance mid-hold", value=False)
            refi_yr = st.number_input("Refi at end of year", 1, int(years) - 1, 3,
                                      step=1, disabled=not refi_on)
            refi_rate = st.number_input("New rate (%)", 0.0, 15.0, 5.5, step=0.125,
                                        disabled=not refi_on)
            refi_term = st.number_input("New term (years)", 5, 40, 30, step=5,
                                        disabled=not refi_on)
            refi_ltv = st.number_input("Refi LTV (%)", 30.0, 90.0, 75.0, step=5.0,
                                       disabled=not refi_on)
            refi_cost = st.number_input("Refi closing costs (% of loan)", 0.0, 6.0,
                                        2.0, step=0.25, disabled=not refi_on)

    assume = Assumptions(
        years=int(years), n_trials=int(n_trials),
        appreciation_mean=app_mean / 100, appreciation_stdev=app_std / 100,
        selling_cost_pct=sell_cost / 100,
        tax_enabled=tax_on, marginal_tax_rate=marg / 100, land_pct=land / 100,
        capital_gains_rate=cg / 100, recapture_rate=recap / 100,
        refi_enabled=refi_on, refi_year=int(refi_yr), refi_rate=refi_rate / 100,
        refi_term_years=int(refi_term), refi_ltv=refi_ltv / 100,
        refi_cost_pct=refi_cost / 100)

    deals = [deal_a] + ([deal_b] if compare else [])
    for d in deals:
        if d.down_payment_pct <= 0:
            st.error(f"{d.label}: down payment must be > 0% — cash-on-cash return "
                     "is undefined with no cash invested.")
            st.stop()

    results = [run_simulation(d, assume) for d in deals]
    sums = [summary_row(r) for r in results]
    labels = [d.label for d in deals]
    yrs = assume.years
    tax_tag = " (after tax)" if assume.tax_enabled else ""

    # ---------------- 1. cash-on-cash
    st.subheader(f"1 · Cash-on-cash return{tax_tag} — expected value and range")
    for lab, s, r in zip(labels, sums, results):
        cols = st.columns(5)
        cols[0].metric(f"{lab} · mean avg CoC", pct(s["coc_mean"]))
        cols[1].metric("10th–90th percentile",
                       f"{pct(s['coc_p10'])} to {pct(s['coc_p90'])}")
        cols[2].metric(f"Mean IRR{tax_tag}", pct(s["irr_mean"]))
        cols[3].metric("IRR 10th–90th",
                       f"{pct(s['irr_p10'])} to {pct(s['irr_p90'])}")
        cols[4].metric("Cash invested", usd(s["cash_invested"]))
    pi_note = " · ".join(f"{l} {usd(s['monthly_payment'])}/mo"
                         for l, s in zip(labels, sums))
    st.caption(md(
        f"P&I {pi_note} · "
        f"exit assumes sale in year {yrs} at the simulated market value "
        f"(appreciation {pct(assume.appreciation_mean)} ± {pct(assume.appreciation_stdev)}/yr), "
        f"net of {pct(assume.selling_cost_pct)} selling costs and the remaining loan balance."
        + (" Refi proceeds are excluded from cash-on-cash and included in IRR."
           if assume.refi_enabled else "")))

    coc_key = "avg_coc_after_tax" if assume.tax_enabled else "avg_coc"
    irr_key = "irr_after_tax" if assume.tax_enabled else "irr"
    st.altair_chart(
        dist_chart([(l, r[coc_key] * 100) for l, r in zip(labels, results)],
                   f"Avg annual cash-on-cash{tax_tag} (%)", alt),
        use_container_width=True)
    irr_series = []
    for l, r in zip(labels, results):
        v = r[irr_key]
        irr_series.append((l, v[np.isfinite(v)] * 100))
    st.altair_chart(
        dist_chart(irr_series, f"{yrs}-yr IRR{tax_tag} (%)", alt),
        use_container_width=True)

    # ---------------- 2. probability of positive cash flow
    st.subheader("2 · Probability of positive cash flow, by year")
    y5 = min(5, yrs)
    for lab, s in zip(labels, sums):
        c1, c2, _ = st.columns([1, 1, 3])
        c1.metric(f"{lab} · Year 1", pct(s["p_pos_y1"]))
        c2.metric(f"Year {y5}", pct(s["p_pos_y5"]))
    prob_df = pd.concat([
        pd.DataFrame({"Year": np.arange(1, yrs + 1),
                      "P": r["p_positive"] * 100, "Deal": l})
        for l, r in zip(labels, results)], ignore_index=True)
    prob_enc = dict(
        x=alt.X("Year:O", axis=alt.Axis(labelAngle=0, title="Year")),
        y=alt.Y("P:Q", scale=alt.Scale(domain=[0, 100]),
                title="Probability of positive cash flow (%)"),
        tooltip=[alt.Tooltip("Deal:N"), alt.Tooltip("Year:O"),
                 alt.Tooltip("P:Q", format=".1f", title="P(positive) %")])
    if len(deals) > 1:
        prob_enc["color"] = alt.Color("Deal:N",
                                      scale=alt.Scale(domain=labels,
                                                      range=[BLUE, ORANGE]),
                                      legend=alt.Legend(title=None, orient="top"))
        prob_enc["xOffset"] = alt.XOffset("Deal:N")
    else:
        prob_enc["color"] = alt.value(BLUE)
    st.altair_chart(
        alt.Chart(prob_df).mark_bar(cornerRadiusTopLeft=4,
                                    cornerRadiusTopRight=4).encode(
            **prob_enc).properties(height=260), use_container_width=True)

    # annual cash flow band, long format, single layer
    cf_frames = []
    for l, r in zip(labels, results):
        cf = r["after_tax_cf"] if assume.tax_enabled else r["annual_cf"]
        for name, vals, band in (
                ("Median", np.percentile(cf, 50, axis=0), "Median"),
                ("10th pct", np.percentile(cf, 10, axis=0), "Range"),
                ("90th pct", np.percentile(cf, 90, axis=0), "Range")):
            cf_frames.append(pd.DataFrame({
                "Year": np.arange(1, yrs + 1), "CF": vals,
                "Series": f"{l} {name}" if len(deals) > 1 else name,
                "Deal": l, "Band": band}))
    st.altair_chart(
        by_year_chart(cf_frames, "CF",
                      f"Annual cash flow{tax_tag} — median and 10th–90th percentile",
                      "Annual cash flow ($)", alt),
        use_container_width=True)

    # ---------------- 3. bad-luck scenario
    st.subheader("3 · Bad-luck scenario — what a rough year costs")
    for lab, d, r, s in zip(labels, deals, results, sums):
        stress = bad_luck_year1(d)
        b = st.columns(3)
        b[0].metric(f"{lab} · expected year 1 cash flow", usd(stress["expected"]))
        b[1].metric("Stress: 3 mo vacant + $12.5k repair", usd(stress["stressed"]),
                    delta=usd(stress["delta"]))
        b[2].metric("Sim: worst year of hold (10th pct)", usd(s["worst_year_p10"]))
    st.caption("Stress case is deterministic: three vacant months plus a mid-range "
               "big repair in year 1, before tax. The sim figure is the 10th "
               "percentile of each trial's single worst year — 10% of simulated "
               "futures had a year at least this bad.")

    # ---------------- 4. DSCR lender view
    st.subheader("4 · DSCR lender view")
    lo_t, hi_t = assume.dscr_thresholds
    for lab, s in zip(labels, sums):
        d1, d2, d3 = st.columns(3)
        d1.metric(f"{lab} · median DSCR (yr 1)", ratio(s["dscr_median"]))
        d2.metric(f"Pass rate @ {lo_t:.2f}", pct(s["dscr_pass_115"]))
        d3.metric(f"Pass rate @ {hi_t:.2f}", pct(s["dscr_pass_125"]))
    dscr_series = []
    for l, r in zip(labels, results):
        v = r["dscr"][:, 0]
        v = v[np.isfinite(v)]
        if v.size:
            dscr_series.append((l, np.clip(v, 0, 3.0)))
    if dscr_series:
        st.altair_chart(dist_chart(dscr_series, "Year-1 DSCR (NOI ÷ debt service)",
                                   alt, fmt=".2f"), use_container_width=True)
    st.caption(f"DSCR uses NOI (collected rent less vacancy, management, "
               f"maintenance, taxes, insurance, HOA) divided by annual debt "
               f"service, excluding big-repair events. Most DSCR lenders want "
               f"{lo_t:.2f}–{hi_t:.2f}+; the pass rate is the share of simulated "
               f"futures clearing that bar in year 1. All-cash deals show as n/a.")

    # ---------------- 5. refi impact
    if assume.refi_enabled:
        st.subheader(f"5 · Refinance at year {assume.refi_year}")
        for lab, r in zip(labels, results):
            cash_out = r["refi_cash"][:, assume.refi_year - 1]
            e1, e2, e3 = st.columns(3)
            e1.metric(f"{lab} · median cash out at refi", usd(np.median(cash_out)))
            e2.metric("10th–90th percentile",
                      f"{usd(np.percentile(cash_out, 10))} to "
                      f"{usd(np.percentile(cash_out, 90))}")
            e3.metric("P(cash-in required)", pct(np.mean(cash_out < 0)))
        st.caption(f"New loan sized at {pct(assume.refi_ltv)} LTV against the "
                   f"simulated market value at year {assume.refi_year}, less the "
                   f"old balance and {pct(assume.refi_cost_pct)} closing costs. "
                   "Negative means the investor brings cash to close — that happens "
                   "when appreciation disappoints.")

    # ---------------- summary + export
    with st.expander("Summary table (all metrics)"):
        for lab, r in zip(labels, results):
            a_cf = r["after_tax_cf"] if assume.tax_enabled else r["annual_cf"]
            rows = {
                f"Avg annual CoC{tax_tag} (%)": r[coc_key] * 100,
                "Year-1 CoC (%)": r["coc"][:, 0] * 100,
                f"{yrs}-yr IRR{tax_tag} (%)": r[irr_key][np.isfinite(r[irr_key])] * 100,
                "Year-1 cash flow ($)": a_cf[:, 0],
                f"Year-{yrs} cash flow ($)": a_cf[:, -1],
                "Year-1 DSCR": r["dscr"][:, 0][np.isfinite(r["dscr"][:, 0])],
                "Value at exit ($)": r["value_by_year"][:, -1],
                "Equity at exit ($)": (r["equity_after_tax"] if assume.tax_enabled
                                       else r["equity_at_exit"]),
            }
            st.markdown(f"**{lab}**" + (f" — {r['inputs'].address}"
                                        if r["inputs"].address else ""))
            st.dataframe(pd.DataFrame({
                "Metric": list(rows),
                "Mean": [np.nanmean(v) if v.size else np.nan for v in rows.values()],
                "10th pct": [np.nanpercentile(v, 10) if v.size else np.nan
                             for v in rows.values()],
                "Median": [np.nanpercentile(v, 50) if v.size else np.nan
                           for v in rows.values()],
                "90th pct": [np.nanpercentile(v, 90) if v.size else np.nan
                             for v in rows.values()],
            }).round(2), hide_index=True, width="stretch")

    st.subheader("Client handout")
    pdf_bytes = build_pdf(results, assume)
    fname = (results[0]["inputs"].address or "rental-deal").strip()
    fname = "".join(ch if ch.isalnum() or ch in "-_ " else "" for ch in fname)
    fname = (fname.replace(" ", "-").lower() or "rental-deal") + "-analysis.pdf"
    st.download_button("Download one-page PDF summary", pdf_bytes, file_name=fname,
                       mime="application/pdf", type="primary")
    st.caption("Branded one-pager with the headline numbers, the probability chart, "
               "the assumptions, and a disclaimer — ready to hand a client.")

    with st.expander("Model assumptions"):
        st.markdown(md(f"""
- **Vacancy** — each month is independently vacant with probability = vacancy %.
- **Big repair** — annual chance drawn from {pct(REPAIR_PROB_RANGE[0])}–{pct(REPAIR_PROB_RANGE[1])} per trial; cost uniform {usd(REPAIR_COST_RANGE[0])}–{usd(REPAIR_COST_RANGE[1])}.
- **Rent growth** — each year drawn Normal(assumption, {pct(RENT_GROWTH_STDEV)} st.dev).
- **Appreciation** — each year drawn Normal({pct(assume.appreciation_mean)}, {pct(assume.appreciation_stdev)}); compounds into the exit value and the refi loan size.
- **Expense inflation** — taxes, insurance, HOA grow {pct(EXPENSE_INFLATION)}/yr.
- **Maintenance** is % of scheduled rent; **management** is % of collected rent.
- **Cash invested** = down payment only (closing costs not modeled).
- **Exit** — sale at simulated market value less {pct(assume.selling_cost_pct)} selling costs and the remaining loan balance.
- **Tax layer** ({'on' if assume.tax_enabled else 'off'}) — straight-line depreciation over {DEPRECIATION_YEARS} years on {pct(1 - assume.land_pct)} of purchase price, mortgage interest and operating expenses deducted, passive losses suspended and carried forward, depreciation recapture at {pct(assume.recapture_rate)} and capital gains at {pct(assume.capital_gains_rate)} on sale. Simplified screening model — **not tax advice**.
- **Refinance** ({'on' if assume.refi_enabled else 'off'}) — new loan at {pct(assume.refi_ltv)} LTV of simulated value at year {assume.refi_year}; proceeds count in IRR, not in cash-on-cash.
- Both deals share one random seed ({assume.seed}), so A and B see the same vacancy months and the same appreciation path — a paired comparison.
"""))


if __name__ == "__main__":
    main()
