"""
Order Smoothing Ratio (OSR).

A measure of how aggressively an ordering policy changes from period to
period. Complementary to BWR (which measures the total variance of
orders): two policies may share the same BWR yet generate very
different period-to-period order churn. A low OSR indicates a policy
that smooths its replenishment decisions, which operationally
corresponds to fewer abrupt production swings and a more stable
supplier relationship.

Definition
----------
OSR_k = Var(O_k(t) - O_k(t-1)) / Var(D(t))
"""

from __future__ import annotations

import numpy as np

from deepbullwhip._types import SimulationResult, TimeSeries
from deepbullwhip.registry import register


@register("metric", "OSR")
class OrderSmoothingRatio:
    """First-difference order variance, normalised by demand variance."""

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = 0,
    ) -> float:
        er = result.echelon_results[echelon]
        orders = np.asarray(er.orders, dtype=float)
        if orders.size < 2:
            return 0.0
        diff = np.diff(orders)
        var_d = float(np.var(demand))
        return float(np.var(diff) / var_d) if var_d > 0 else 0.0
