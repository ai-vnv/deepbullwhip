from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

TimeSeries = NDArray[np.float64]


@dataclass
class EchelonResult:
    """Results for a single echelon after simulation."""

    name: str
    orders: TimeSeries
    inventory_levels: TimeSeries
    costs: TimeSeries
    bullwhip_ratio: float
    fill_rate: float
    total_cost: float


@dataclass
class SimulationResult:
    """Aggregate results from a full supply chain simulation."""

    echelon_results: list[EchelonResult]
    cumulative_bullwhip: float
    total_cost: float

    def to_dict(self) -> dict[str, float]:
        """Flat dictionary matching the notebook's get_metrics() output."""
        d: dict[str, float] = {}
        for k, er in enumerate(self.echelon_results):
            d[f"BW_{k + 1}"] = er.bullwhip_ratio
            d[f"cost_{k + 1}"] = er.total_cost
            d[f"fill_rate_{k + 1}"] = er.fill_rate
        d["BW_cumulative"] = self.cumulative_bullwhip
        d["total_cost"] = self.total_cost
        return d
