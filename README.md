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

Run a single ticker/date:

```
python -m analysis.run --data_root data/raw --ticker AMZN --date 2012-06-21
```

Run all discovered pairs:

```
python -m analysis.run --data_root data/raw --all
```

Outputs are written under `results/` (processed datasets, features, plots,
reports, and sanity checks).
