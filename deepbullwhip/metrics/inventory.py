"""Inventory-related metrics."""

import numpy as np

from deepbullwhip._types import SimulationResult, TimeSeries
from deepbullwhip.registry import register


@register("metric", "NSAmp")
class NSAmp:
    """Net Stock Amplification: Var(Inventory) / Var(Demand).

    Disney & Towill (2003) analyze the BWR-NSAmp tradeoff.
    """

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = 0,
    ) -> float:
        er = result.echelon_results[echelon]
        var_d = np.var(demand)
        return float(np.var(er.inventory_levels) / var_d) if var_d > 0 else 0.0


@register("metric", "FILL_RATE")
class FillRate:
    """Fill Rate: fraction of periods with non-negative inventory."""

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = 0,
    ) -> float:
        er = result.echelon_results[echelon]
        return float(np.mean(er.inventory_levels >= 0))
