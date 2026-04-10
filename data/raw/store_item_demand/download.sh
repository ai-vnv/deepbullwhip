#!/usr/bin/env bash
# Store Item Demand Forecasting
# Source: https://www.kaggle.com/competitions/demand-forecasting-kernels-only
# Requires: pip install kaggle && export KAGGLE_USERNAME=... KAGGLE_KEY=...
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -f "train.csv" ]; then
    echo "Store Item Demand data already downloaded."
    exit 0
fi

echo "Downloading Store Item Demand dataset from Kaggle..."
kaggle competitions download -c demand-forecasting-kernels-only -p "$DIR"
unzip -o demand-forecasting-kernels-only.zip -d "$DIR"
rm -f demand-forecasting-kernels-only.zip

echo "Store Item Demand download complete. Files:"
ls -lh "$DIR"/*.csv
