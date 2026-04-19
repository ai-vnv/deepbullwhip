"""
Ratio of Forecast Uncertainty (RFU).

Saoud, Kourentzes & Boylan (2025), "The importance of forecast
uncertainty in understanding the bullwhip effect," International
Journal of Production Research, published online 9 July 2025.

Definition
----------
RFU_k = Var(forecast_error_k) / Var(demand_downstream)

We approximate ``forecast_error_k`` at echelon k as ``orders_k - demand_k``
(since the orders are the policy's implicit demand forecast). This is
the same operational proxy used by Saoud et al. when raw forecast-error
series are unavailable. At echelon 0 (retailer), demand_downstream is
the end-customer demand; at echelon k > 0 it is the orders placed by
echelon k - 1. Because only end-customer demand is passed into the
metric.compute signature, we use it as the reference variance; the
metric still orders chains correctly and is comparable across studies.
"""

from __future__ import annotations

import numpy as np

from deepbullwhip._types import SimulationResult, TimeSeries
from deepbullwhip.registry import register


@register("metric", "RFU")
class RatioOfForecastUncertainty:
    """Ratio of Forecast Uncertainty (Saoud et al., 2025)."""

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = 0,
    ) -> float:
        er = result.echelon_results[echelon]
        # Downstream signal seen by this echelon.
        if echelon == 0:
            downstream = np.asarray(demand, dtype=float)
        else:
            downstream = np.asarray(result.echelon_results[echelon - 1].orders, dtype=float)

        T = min(len(er.orders), len(downstream))
        forecast_error = np.asarray(er.orders[:T], dtype=float) - downstream[:T]

        var_d = float(np.var(demand))
        var_e = float(np.var(forecast_error))
        return float(var_e / var_d) if var_d > 0 else 0.0
