"""
Inventory Turnover.

A classical operations-management KPI. High turnover (small average
on-hand inventory relative to demand) indicates working-capital
efficiency; very high turnover coupled with low fill rate indicates
under-stocking. Reported here as an annualised ratio assuming the
period corresponds to weekly data (factor 52); override via the
``InventoryTurnover.periods_per_year`` class attribute.

Definition
----------
Turnover_k = periods_per_year * mean(D) / mean( max(I_k, 0) )
"""

from __future__ import annotations

import numpy as np

from deepbullwhip._types import SimulationResult, TimeSeries
from deepbullwhip.registry import register


@register("metric", "InventoryTurnover")
class InventoryTurnover:
    """Annualised inventory turnover."""

    periods_per_year: int = 52

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = 0,
    ) -> float:
        er = result.echelon_results[echelon]
        inv = np.asarray(er.inventory_levels, dtype=float)
        mean_on_hand = float(np.mean(np.maximum(inv, 0.0)))
        mean_demand = float(np.mean(demand))
        if mean_on_hand <= 1e-9:
            return float("inf")
        return float(InventoryTurnover.periods_per_year * mean_demand / mean_on_hand)
