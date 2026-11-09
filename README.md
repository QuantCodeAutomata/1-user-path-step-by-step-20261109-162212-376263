# SmartDelta — user path step by step

Replication of the three reproducible components described in
**SmartDelta — "User path step by step"** (product specification document):

1. a Monte Carlo pricing engine for Synthetic and Autocall structured products
   (Sections 1.7–1.11 and Section 4);
2. a historical backtest with stress testing and savings-benchmark comparison
   (Sections 1.12 and 4);
3. a dynamic delta-hedging and fractional-rebalancing simulation with full
   execution frictions (Sections 1.23–1.28 and Section 4).

## Layout

```
src/              Shared library (config, pricing, backtest, hedging, rates, market)
data/             Market-data loader and synthetic-data generator
exp/              Experiment scripts: exp1_pricing, exp2_backtest, exp3_hedging
tests/            Pytest suite (unit tests + optional market-data tests)
results/          Generated tables, figures and RESULTS.md
```

Python modules under `src/` are flat and contain no subdirectories. Experiments
are run from the repository root as `python -m exp.<name>`; no `sys.path`
mutations are needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

* `DATALOADER_API_KEY` — QuantX Dataloader; obtain at <https://quantx.limex.pro/>.
* `MASSIVE_API_KEY`    — Massive REST API; obtain at <https://massive.com/>.
* `FRED_API_KEY` (optional) — Treasury-curve downloads via FRED.

Secrets live only in the uncommitted `.env`; they must never enter source,
configuration, logs or commits.

## Tests

```bash
python -m pytest                       # all unit tests
python -m pytest -m "not market_data"  # skip live API tests
```

Unit tests rely on synthetic fixtures. The optional `market_data` tests load a
short reduced slice through `data/data_loader.py` and are skipped when
`DATALOADER_API_KEY` is absent.

## Experiments

Each experiment overwrites the files it owns inside `results/`. After every run
inspect `results/RESULTS.md` and update verdicts by hand against the freshly
generated numbers.

```bash
python -m exp.exp1_pricing              # reduced path count for runtime
python -m exp.exp1_pricing --full       # 100,000 Monte Carlo paths (Section 4)
python -m exp.exp2_backtest             # historical backtest + stress scenarios
python -m exp.exp3_hedging              # delta-hedging simulation
```

Convenience entry point:

```bash
bash reproduce.sh
```

### Full paper configuration

The paper specifies **100,000 Monte Carlo paths** (Section 4) and the platform
supports **up to 10 underlyings**. The default run uses 50,000 paths over 5
underlyings; `--full` restores the paper's 100,000-path configuration. The
universe size is set in `src/config.UNIVERSE` and may be expanded up to 10
tickers.

## Conventions

* All parameters live in `src/config.py` as UPPER_CASE module-level variables,
  each with a comment giving its meaning, unit and the paper section it
  originates from (or the note "not specified in the paper").
* Market data enters the repository only through `data/data_loader.py`.
* Dynamic-hedging quantities are carried as `decimal.Decimal` and rounded to
  8 decimal places (Section 4 "8 decimal places").
* No downloaded data or caches are ever committed; `data/cache/` is in
  `.gitignore`.

## Paper

Document: *SmartDelta — "1. User path step by step"*. The product/platform
specification used as the source of truth is a single self-contained document
with no external references. Section numbers in the code refer to that
document.
