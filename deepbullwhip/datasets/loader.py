"""Unified loader for raw downloaded datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deepbullwhip._types import TimeSeries

_DATA_ROOT = Path(__file__).parents[2] / "data" / "raw"


def load_dataset(
    name: str,
    freq: str = "weekly",
    data_root: Path | str | None = None,
    **kwargs,
) -> TimeSeries:
    """Load a downloaded dataset as a 1-D demand time series.

    Parameters
    ----------
    name : str
        Dataset name: "m5", "uci_online_retail", "store_item_demand",
        "australian_drug_sales".
    freq : str
        Aggregation frequency: "daily", "weekly", "monthly".
    data_root : Path or None
        Override the default data/raw directory.
    **kwargs
        Dataset-specific parameters (see individual loaders).

    Returns
    -------
    TimeSeries, shape (T,)
    """
    root = Path(data_root) if data_root else _DATA_ROOT
    loaders = {
        "m5": _load_m5,
        "uci_online_retail": _load_uci_online_retail,
        "store_item_demand": _load_store_item_demand,
        "australian_drug_sales": _load_australian_drug_sales,
    }
    if name not in loaders:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {list(loaders.keys())}"
        )
    return loaders[name](root / name, freq=freq, **kwargs)


def _load_m5(
    data_dir: Path,
    freq: str = "weekly",
    store: str = "CA_1",
    dept: str = "FOODS_1",
) -> TimeSeries:
    """Load M5 Walmart sales data.

    Parameters
    ----------
    store : str
        Store ID, e.g. "CA_1", "TX_1", "WI_1".
    dept : str
        Department, e.g. "FOODS_1", "FOODS_2", "FOODS_3",
        "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2".
    freq : str
        "daily" or "weekly".
    """
    csv_path = data_dir / "sales_train_evaluation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"M5 data not found at {csv_path}. "
            f"Run: bash data/raw/m5/download.sh"
        )
    df = pd.read_csv(csv_path)
    mask = df["store_id"].eq(store) & df["dept_id"].eq(dept)
    day_cols = [c for c in df.columns if c.startswith("d_")]
    daily = df.loc[mask, day_cols].sum(axis=0).values.astype(np.float64)

    if freq == "weekly":
        n_weeks = len(daily) // 7
        return daily[: n_weeks * 7].reshape(n_weeks, 7).sum(axis=1)
    elif freq == "monthly":
        n_months = len(daily) // 30
        return daily[: n_months * 30].reshape(n_months, 30).sum(axis=1)
    return daily


def _load_uci_online_retail(
    data_dir: Path,
    freq: str = "weekly",
    country: str = "United Kingdom",
    top_n: int = 1,
) -> TimeSeries:
    """Load UCI Online Retail transactional data as aggregated demand.

    Parameters
    ----------
    country : str
        Filter by country (default "United Kingdom").
    top_n : int
        Use top N products by total quantity (default 1 = highest-volume).
    freq : str
        "daily", "weekly", or "monthly".
    """
    xlsx_path = data_dir / "online_retail.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"UCI Online Retail data not found at {xlsx_path}. "
            f"Run: bash data/raw/uci_online_retail/download.sh"
        )
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    if country:
        df = df[df["Country"] == country]

    # Find top products by total quantity
    top_products = (
        df.groupby("StockCode")["Quantity"]
        .sum()
        .nlargest(top_n)
        .index
    )
    df = df[df["StockCode"].isin(top_products)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.set_index("InvoiceDate")

    freq_map = {"daily": "D", "weekly": "W", "monthly": "ME"}
    agg_freq = freq_map.get(freq, "W")
    demand = df["Quantity"].resample(agg_freq).sum().fillna(0)

    return demand.values.astype(np.float64)


def _load_store_item_demand(
    data_dir: Path,
    freq: str = "weekly",
    store: int = 1,
    item: int = 1,
) -> TimeSeries:
    """Load Store Item Demand Forecasting data.

    Parameters
    ----------
    store : int
        Store number (1-10).
    item : int
        Item number (1-50).
    freq : str
        "daily", "weekly", or "monthly".
    """
    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Store Item Demand data not found at {csv_path}. "
            f"Run: bash data/raw/store_item_demand/download.sh"
        )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df[(df["store"] == store) & (df["item"] == item)]
    df = df.set_index("date").sort_index()

    if freq == "daily":
        return df["sales"].values.astype(np.float64)

    freq_map = {"weekly": "W", "monthly": "ME"}
    agg_freq = freq_map.get(freq, "W")
    demand = df["sales"].resample(agg_freq).sum().fillna(0)
    return demand.values.astype(np.float64)


def _load_australian_drug_sales(
    data_dir: Path,
    freq: str = "monthly",
    atc2: str = "A10",
    type_: str = "CONCESSIONAL SAFETY NET",
) -> TimeSeries:
    """Load Australian PBS pharmaceutical drug sales.

    The raw CSV from tidyverts/tsibbledata is wide-format:
    Col 0 = Type, Col 1 = ATC2, Col 2 = Description, Col 3+ = monthly scripts.

    Parameters
    ----------
    atc2 : str
        ATC2 drug classification code, e.g. "A10" (antidiabetic therapy),
        "A02" (acid-related disorders), "J01" (antibiotics), "R06" (antihistamines).
    type_ : str
        One of "CONCESSIONAL SAFETY NET", "CO-PAYMENTS", "GENERAL SAFETY NET",
        "GENERAL CO-PAYMENTS", etc.
    freq : str
        "monthly" (native) or "quarterly".
    """
    csv_path = data_dir / "PBS.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Australian Drug Sales data not found at {csv_path}. "
            f"Run: bash data/raw/australian_drug_sales/download.sh"
        )
    df = pd.read_csv(csv_path, header=None)

    # Find data rows: col0=type, col1=atc2 code, col2=description, col3+=values
    # Skip header rows (first ~6 rows are metadata)
    data_rows = df[df.iloc[:, 1].str.match(r"^[A-Z]\d{2}$", na=False)].copy()

    type_upper = type_.upper()
    mask = (
        data_rows.iloc[:, 0].str.upper().str.strip().eq(type_upper)
        & data_rows.iloc[:, 1].str.upper().str.strip().eq(atc2.upper())
    )
    matched = data_rows.loc[mask]

    if matched.empty:
        available_types = sorted(data_rows.iloc[:, 0].dropna().unique())
        available_atc2 = sorted(data_rows.iloc[:, 1].dropna().unique())
        raise KeyError(
            f"No data for ATC2='{atc2}', Type='{type_}'. "
            f"Available Types: {available_types[:5]}... "
            f"Available ATC2: {available_atc2[:10]}..."
        )

    # Extract monthly values (columns 3 onward)
    values = matched.iloc[0, 3:].values
    demand = pd.to_numeric(values, errors="coerce")
    demand = demand[~np.isnan(demand)].astype(np.float64)

    if freq == "quarterly":
        n_q = len(demand) // 3
        demand = demand[: n_q * 3].reshape(n_q, 3).sum(axis=1)

    return demand


def list_datasets() -> list[str]:
    """Return names of all supported datasets."""
    return ["m5", "uci_online_retail", "store_item_demand", "australian_drug_sales"]
