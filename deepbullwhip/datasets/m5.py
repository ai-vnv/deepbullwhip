"""M5 Walmart demand dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deepbullwhip._types import TimeSeries

CACHE_DIR = Path.home() / ".deepbullwhip" / "data" / "m5"


def load_m5(
    store: str = "CA_1",
    dept: str = "FOODS_1",
    freq: str = "weekly",
    cache_dir: Path = CACHE_DIR,
) -> TimeSeries:
    """Load M5 Walmart demand data.

    On first call, downloads from Kaggle (requires ``kaggle`` package
    and KAGGLE_USERNAME / KAGGLE_KEY environment variables). Caches
    locally for subsequent calls.

    Parameters
    ----------
    store : str
        Store identifier, e.g. "CA_1", "TX_1".
    dept : str
        Department, e.g. "FOODS_1", "HOBBIES_1".
    freq : str
        "daily" or "weekly" (weekly aggregates daily to 7-day sums).
    cache_dir : Path
        Local cache directory.

    Returns
    -------
    TimeSeries, shape (T,)
    """
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"m5_{store}_{dept}_{freq}.npy"
    if cache_file.exists():
        return np.load(cache_file)

    # Download via Kaggle API
    try:
        import kaggle  # noqa: F401

        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            "m5-forecasting-accuracy", path=str(cache_dir)
        )
    except ImportError:
        raise ImportError(
            "M5 download requires the kaggle package. "
            "Install with: pip install kaggle\n"
            "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
            "Or download manually from "
            "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"
        ) from None

    # Parse the downloaded CSV
    import zipfile

    zip_path = cache_dir / "m5-forecasting-accuracy.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    sales_path = cache_dir / "sales_train_evaluation.csv"
    df = pd.read_csv(sales_path)

    # Filter to store/dept
    mask = df["store_id"].eq(store) & df["dept_id"].eq(dept)
    day_cols = [c for c in df.columns if c.startswith("d_")]
    demand_daily = df.loc[mask, day_cols].sum(axis=0).values.astype(np.float64)

    if freq == "weekly":
        n_weeks = len(demand_daily) // 7
        demand = demand_daily[: n_weeks * 7].reshape(n_weeks, 7).sum(axis=1)
    else:
        demand = demand_daily

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, demand)
    return demand
