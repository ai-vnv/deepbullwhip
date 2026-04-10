#!/usr/bin/env bash
# Australian Pharmaceutical Drug Sales (PBS)
# Source: tidyverts/tsibbledata R package
# No credentials required.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -f "PBS.csv" ] && [ "$(wc -l < PBS.csv)" -gt 10 ]; then
    echo "Australian Drug Sales data already downloaded."
    exit 0
fi

echo "Downloading Australian PBS drug sales data..."
curl -L -o PBS.csv \
    "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/PBS/PBS.csv"

LINES=$(wc -l < PBS.csv)
if [ "$LINES" -lt 10 ]; then
    echo "ERROR: Download failed (file too small: $LINES lines)."
    rm -f PBS.csv
    exit 1
fi

echo "Australian Drug Sales download complete ($LINES lines)."
