"""
Peak Bullwhip Ratio.

The standard BWR averages over the full horizon and therefore can hide
acute panic-ordering spikes that are operationally the most damaging
(capacity booking, overtime, expediting). The Peak BWR complements BWR
by reporting the worst-case over a rolling window.

Definition
----------
PeakBWR_k = max_{t in [w, T]} Var_{s in [t-w, t]}(O_k(s))
                               / Var_{s in [t-w, t]}(D(s))

The default window is 26 periods (half a year at weekly cadence); it
can be overridden via the ``PeakBWR.window`` class attribute, e.g.::

    from deepbullwhip.metrics.peak_bwr import PeakBWR
    PeakBWR.window = 13
"""

from __future__ import annotations

import numpy as np

from deepbullwhip._types import SimulationResult, TimeSeries
from deepbullwhip.registry import register


@register("metric", "PeakBWR")
class PeakBWR:
    """Maximum rolling-window bullwhip ratio."""

    # Class-level configuration (overridable from user code).
    window: int = 26

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = 0,
    ) -> float:
        er = result.echelon_results[echelon]
        orders = np.asarray(er.orders, dtype=float)
        dem = np.asarray(demand, dtype=float)

        w = min(PeakBWR.window, orders.size, dem.size)
        if w < 4:
            var_d = float(np.var(dem))
            return float(np.var(orders) / var_d) if var_d > 0 else 0.0

        best = 0.0
        for start in range(0, orders.size - w + 1):
            o_win = orders[start : start + w]
            d_win = dem[start : start + w] if dem.size >= start + w else dem[-w:]
            v_d = float(np.var(d_win))
            if v_d <= 0:
                continue
            ratio = float(np.var(o_win) / v_d)
            if ratio > best:
                best = ratio
        return best
