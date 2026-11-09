#!/usr/bin/env bash
# Run the full test suite and every experiment, writing all artefacts into
# results/. Invoke from the repository root.
set -euo pipefail

python -m pytest -m "not market_data"
python -m exp.exp1_pricing
python -m exp.exp2_backtest
python -m exp.exp3_hedging
