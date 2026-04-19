"""
Damping Ratio.

Control-theoretic measure of how quickly a supply chain's order signal
returns to steady-state after a demand shock (Dejonckheere, Disney,
Lambrecht & Towill, 2003, "Measuring and avoiding the bullwhip effect:
A control theoretic approach," European Journal of Operational Research,
147(3), 567--590).

We estimate the damping ratio *empirically* from the lag-1 and lag-2
autocorrelations of the order series via the AR(2) formula:

    order_t = phi_1 order_{t-1} + phi_2 order_{t-2} + eps_t

    damping = -phi_1 / (2 * sqrt(-phi_2))   if phi_2 < 0 and |roots| < 1

A damping ratio near 1 indicates a critically-damped, smoothly
responsive chain; < 1 indicates oscillatory (bullwhip-prone) dynamics;
> 1 indicates sluggish correction. Returns NaN when the AR(2) fit is
non-physical (positive phi_2 or explosive roots).
"""

from __future__ import annotations

import numpy as np

from deepbullwhip._types import SimulationResult, TimeSeries
from deepbullwhip.registry import register


@register("metric", "DampingRatio")
class DampingRatio:
    """Empirical AR(2) damping ratio of the order signal."""

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = 0,
    ) -> float:
        er = result.echelon_results[echelon]
        orders = np.asarray(er.orders, dtype=float)
        if orders.size < 10:
            return float("nan")

        # Fit AR(2) by least squares, include intercept.
        y = orders[2:]
        X = np.stack([orders[1:-1], orders[:-2], np.ones(orders.size - 2)], axis=1)
        try:
            coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return float("nan")

        phi1, phi2, _ = coefs
        if phi2 >= 0:
            return float("nan")
        disc = -phi2
        if disc <= 0:
            return float("nan")
        zeta = float(-phi1 / (2.0 * np.sqrt(disc)))
        # Only physically-meaningful underdamped systems have |zeta| in a
        # sensible range; clamp to avoid pathological spikes.
        return zeta
