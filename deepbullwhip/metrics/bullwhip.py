"""Bullwhip ratio metrics."""

import numpy as np

from deepbullwhip._types import SimulationResult, TimeSeries
from deepbullwhip.registry import register


@register("metric", "BWR")
class BWR:
    """Bullwhip Ratio: Var(Orders) / Var(Demand)."""

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = 0,
    ) -> float:
        er = result.echelon_results[echelon]
        var_d = np.var(demand)
        return float(np.var(er.orders) / var_d) if var_d > 0 else 0.0


@register("metric", "CUM_BWR")
class CumulativeBWR:
    """Cumulative Bullwhip Ratio: Var(last echelon orders) / Var(Demand)."""

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = -1,
    ) -> float:
        # echelon parameter is ignored; uses last echelon by definition
        last_er = result.echelon_results[-1]
        var_d = np.var(demand)
        return float(np.var(last_er.orders) / var_d) if var_d > 0 else 0.0
