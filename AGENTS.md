# AGENTS.md — repository guide

Replication of **SmartDelta — "1. User path step by step"**:
pricing, backtest and dynamic-hedging components described by the product
specification document.

## Layout

```
README.md               Setup, test and run commands
reproduce.sh            Runs tests + every experiment
requirements.txt
.env.example            Empty placeholders for DATALOADER/MASSIVE API keys
.gitignore              Keeps data/cache/, .env, __pycache__ etc. out of git
src/                    Flat library modules
  config.py             All experiment parameters (UPPER_CASE)
  market.py             Spot/volatility/correlation/dividend estimation
  rates.py              Flat and Treasury-bootstrapped yield curves
  pricing.py            Monte Carlo pricer: Synthetic + Autocall + Barriers
  backtest.py           Walk-forward backtest, stress, tracking error, VaR...
  hedging.py            Decimal-precision delta hedging engine
data/                   Market-data access only; never commits downloads
  data_loader.py        QuantX Dataloader REST wrapper (OHLCV)
  generate_synthetic.py Synthetic price paths for tests and demos
exp/                    Experiment scripts (python -m exp.<name>)
  exp1_pricing.py
  exp2_backtest.py
  exp3_hedging.py
tests/                  Pytest unit + optional market-data tests
results/                Artefacts + RESULTS.md (verdicts and comparisons)
```

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in DATALOADER_API_KEY, MASSIVE_API_KEY

# Tests
python -m pytest                       # all
python -m pytest -m "not market_data"  # no live API calls

# Experiments (write into results/)
python -m exp.exp1_pricing              # 50,000 MC paths (reduced)
python -m exp.exp1_pricing --full       # 100,000 MC paths (Section 4)
python -m exp.exp2_backtest
python -m exp.exp3_hedging

# One-shot
bash reproduce.sh
```

## Conventions

* All parameters live in `src/config.py` as UPPER_CASE module-level variables
  with a comment giving meaning, unit and the paper section.
* Market data enters only through `data/data_loader.py`.
* Hedging quantities are `decimal.Decimal` at 8 decimal places (Section 4).
* Secrets live only in the uncommitted `.env`.
* Downloaded market data is never committed. `data/cache/` is ignored.
* After rerunning an experiment, check and update the verdicts in
  `results/RESULTS.md` against the freshly saved files.

## Tests

* Unit tests use synthetic data from `data/generate_synthetic.py`.
* `@pytest.mark.market_data` tests load a short slice through the real
  Dataloader and are skipped when `DATALOADER_API_KEY` is unset.
