#!/usr/bin/env bash
# M5 Forecasting — Walmart store sales
# Source: https://www.kaggle.com/competitions/m5-forecasting-accuracy
# Requires: pip install kaggle && export KAGGLE_USERNAME=... KAGGLE_KEY=...
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -f "sales_train_evaluation.csv" ]; then
    echo "M5 data already downloaded."
    exit 0
fi

echo "Downloading M5 dataset from Kaggle..."
kaggle competitions download -c m5-forecasting-accuracy -p "$DIR"
unzip -o m5-forecasting-accuracy.zip -d "$DIR"
rm -f m5-forecasting-accuracy.zip

echo "M5 download complete. Files:"
ls -lh "$DIR"/*.csv
