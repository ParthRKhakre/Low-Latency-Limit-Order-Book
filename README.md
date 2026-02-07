# Low-Latency Limit Order Book Market Maker

## Dataset layout and autodiscovery
Place LOBSTER CSV pairs under `data/raw/<TICKER>/<YYYY-MM-DD>/` as:

```
message.csv
orderbook.csv
```

The analysis runner will auto-discover all pairs beneath `data/raw` and also
fallback-scan the repository for files matching `*_message*.csv` and
`*_orderbook*.csv` if your data lives elsewhere. When discovered outside the
expected folder structure, it will attempt to infer `<TICKER>` and `<DATE>`
from the filename pattern.

## Quick start
Install dependencies:

```
pip install -r requirements.txt
```

The notebook reads optional environment variables:
- `LOB_DATA_ROOT` (default: `data/raw`)
- `LOB_OUT` (default: `results`)
- `LOB_TICKER` and `LOB_DATE` (required unless `LOB_ALL=1`)
- `LOB_ALL=1` to run all discovered pairs
- `LOB_FORMAT` (`parquet` or `csv`)
- `LOB_NO_PLOTS=1` to skip plotting

Run a single ticker/date (executes the notebook):

```
LOB_TICKER=AMZN LOB_DATE=2012-06-21 jupyter nbconvert --execute --to notebook --inplace analysis/lobster_pipeline.ipynb
```

Run all discovered pairs:

```
LOB_ALL=1 jupyter nbconvert --execute --to notebook --inplace analysis/lobster_pipeline.ipynb
```

Outputs are written under `results/` (processed datasets, features, plots,
reports, and sanity checks).
