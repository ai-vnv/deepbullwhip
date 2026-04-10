"""WSTS semiconductor billings sample data loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deepbullwhip._types import TimeSeries

_DATA_DIR = Path(__file__).parent / "data"


def load_wsts(
    region: str = "worldwide",
    product: str = "total",
) -> TimeSeries:
    """Load WSTS semiconductor billings sample data.

    A small bundled sample (5 years monthly) is included in the package.
    For the full 40-year dataset, download from:
    https://www.wsts.org/67/Historical-Billings-Report

    Parameters
    ----------
    region : str
        One of "worldwide", "americas", "europe", "japan", "asia_pacific".
    product : str
        One of "total", "discrete", "optoelectronics", "analog", "logic",
        "memory", "micro", "sensors".

    Returns
    -------
    TimeSeries, shape (T,)
    """
    data_path = _DATA_DIR / "wsts_sample.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"WSTS sample data not found at {data_path}. "
            "The bundled sample may not have been installed correctly."
        )

    df = pd.read_csv(data_path)
    col = f"{region}_{product}"
    if col not in df.columns:
        available = [c for c in df.columns if c != "date"]
        raise KeyError(
            f"Column '{col}' not found. Available: {available}"
        )

    return df[col].values.astype(np.float64)
