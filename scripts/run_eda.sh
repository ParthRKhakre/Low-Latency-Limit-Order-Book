#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-data/raw}
TICKER=${2:-AMZN}
DATE=${3:-2012-06-21}

LOB_DATA_ROOT="${DATA_ROOT}" LOB_TICKER="${TICKER}" LOB_DATE="${DATE}" \
  jupyter nbconvert --execute --to notebook --inplace analysis/lobster_pipeline.ipynb
