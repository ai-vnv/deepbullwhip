#!/usr/bin/env bash
# UCI Online Retail Dataset (ID 352)
# Source: https://archive.ics.uci.edu/dataset/352/online+retail
# No credentials required.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -f "online_retail.xlsx" ] && [ "$(stat -f%z online_retail.xlsx 2>/dev/null || stat --format=%s online_retail.xlsx 2>/dev/null)" -gt 1000 ]; then
    echo "UCI Online Retail data already downloaded."
    exit 0
fi

echo "Downloading UCI Online Retail dataset..."
# Primary: UCI static archive
curl -L --fail -o online_retail.zip \
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip" 2>/dev/null && {
    unzip -o online_retail.zip -d "$DIR"
    rm -f online_retail.zip
    # Rename to consistent name
    if [ -f "Online Retail.xlsx" ]; then
        mv "Online Retail.xlsx" online_retail.xlsx
    fi
    echo "UCI Online Retail download complete."
    exit 0
}

# Fallback: direct xlsx from old URL
echo "Primary URL failed, trying fallback..."
curl -L --fail -o online_retail.xlsx \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx" 2>/dev/null && {
    echo "UCI Online Retail download complete (fallback)."
    exit 0
}

echo "ERROR: Could not download UCI Online Retail. UCI archive may be down."
echo "Try manually: https://archive.ics.uci.edu/dataset/352/online+retail"
echo "Or Kaggle mirror: https://www.kaggle.com/datasets/jihyeseo/online-retail-data-set-from-uci-ml-repo"
exit 1
