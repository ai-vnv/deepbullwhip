# Datasets

Raw datasets for benchmarking supply chain simulations. Each subdirectory contains a `download.sh` script to fetch from the original source.

## Available Datasets

| Dataset | Source | Size | Frequency | Description |
|---------|--------|------|-----------|-------------|
| **M5** | [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy) | ~500MB | Daily | Walmart store sales (3049 products, 10 stores, 1941 days) |
| **UCI Online Retail** | [UCI ML Repo](https://archive.ics.uci.edu/dataset/352/online+retail) | ~45MB | Transactional | UK online retail invoices (Dec 2010 - Dec 2011) |
| **Store Item Demand** | [Kaggle](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) | ~3MB | Daily | 50 items across 10 stores, 5 years |
| **Australian Drug Sales** | [R `fpp3` package](https://pkg.robjhyndman.com/fpp3package/) | ~50KB | Monthly | PBS pharmaceutical subsidies by drug type (1991-2008) |
| **Beer Game** | Built-in | N/A | Weekly | Classic MIT Beer Game step demand (deterministic) |
| **WSTS** | Bundled sample / [WSTS.org](https://www.wsts.org) | ~5KB | Monthly | Semiconductor billings by region/product (2019-2023 sample) |

## Download Instructions

```bash
# Download all (requires kaggle CLI for M5 and Store Item)
for d in data/raw/*/; do
    if [ -f "$d/download.sh" ]; then
        echo "Downloading $(basename $d)..."
        bash "$d/download.sh"
    fi
done
```

## Usage with DeepBullwhip

```python
from deepbullwhip.demand.replay import ReplayDemandGenerator
from deepbullwhip.datasets.loader import load_dataset
import numpy as np

# Load any downloaded dataset as a demand series
demand = load_dataset("uci_online_retail", freq="weekly", product="top1")

# Use in benchmarks via ReplayDemandGenerator
gen = ReplayDemandGenerator(data=demand)
```
