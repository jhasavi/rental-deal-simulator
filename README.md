# Rental Deal Simulator

Monte Carlo underwriting for rental / investor deals — distributions, not
single numbers. Namaste Boston Homes.

## Setup

The repo ships no virtual environment. Create one once per machine:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
```

(`python3` needs to be 3.10+. On a Mac where the system/Homebrew `python3` is
too new or missing `venv` support cleanly, point at a specific interpreter
instead, e.g. `python3.13 -m venv .venv`.)

## Run

```bash
.venv/bin/streamlit run rental_deal_simulator.py
```

Opens at `http://localhost:8501`. The **Where to buy** panel at the top ranks
Massachusetts towns; the **Add a second unit** panel underwrites a by-right
ADU; the sidebar's **Assumptions** tab is where a DSCR loan is turned on.

## Test

```bash
.venv/bin/python -m pytest -q
```

All modules are pure Python/NumPy underneath the Streamlit UI, so the whole
suite runs without a browser or a running server.

## Layout

| File | What it is |
| --- | --- |
| `rental_deal_simulator.py` | The app. Vectorized Monte Carlo core + Streamlit UI, importable without Streamlit for testing. |
| `dscr.py` | DSCR loan sizing — see [DSCR_PLAYBOOK.md](DSCR_PLAYBOOK.md). |
| `adu.py` | Accessory dwelling unit economics — see [ADU_PLAYBOOK.md](ADU_PLAYBOOK.md). |
| `town_screener.py`, `market_data.py` | Massachusetts town ranking — see [WHERE_TO_BUY.md](WHERE_TO_BUY.md). |
| `test_*.py` | Pytest suites, one per module above. |

## Data refresh

Town-level market data (Zillow ZHVI/ZORI/inventory) is downloaded, not
committed:

```bash
.venv/bin/python market_data.py
```

Cached under `data/market/` (gitignored) so it doesn't go stale silently in
the repo.
