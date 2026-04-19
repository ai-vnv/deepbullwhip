"""
Expected Shortfall (ES).

The average magnitude of backorders conditional on a stockout period.
While FillRate captures the probability of a stockout, ES captures its
*severity* and is therefore the relevant quantity for service-level
agreements tied to missed-units penalties.

Definition
----------
ES_k = mean( max(-I_k(t), 0) | I_k(t) < 0 )

Returns 0 if no stockouts occur.
"""

from __future__ import annotations

import numpy as np

from deepbullwhip._types import SimulationResult, TimeSeries
from deepbullwhip.registry import register


@register("metric", "ExpectedShortfall")
class ExpectedShortfall:
    """Average backorder magnitude conditional on a stockout."""

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = 0,
    ) -> float:
        er = result.echelon_results[echelon]
        inv = np.asarray(er.inventory_levels, dtype=float)
        backorders = np.maximum(-inv, 0.0)
        stockout_mask = inv < 0
        if not np.any(stockout_mask):
            return 0.0
        return float(np.mean(backorders[stockout_mask]))
