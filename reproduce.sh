#!/usr/bin/env bash
# Runs the test suite and both experiments in their executed configuration.
set -uo pipefail

cd "$(dirname "$0")"

python -m pytest -m "not market_data"
python -m exp.exp1
python -m exp.exp2
