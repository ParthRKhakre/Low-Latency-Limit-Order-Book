#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-data/raw}
TICKER=${2:-AMZN}
DATE=${3:-2012-06-21}

python -m analysis.run --data_root "${DATA_ROOT}" --ticker "${TICKER}" --date "${DATE}"
