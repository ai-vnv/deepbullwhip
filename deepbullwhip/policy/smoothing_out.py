"""Smoothing Order-Up-To policy with exponential order smoothing."""

import numpy as np
from scipy import stats

from deepbullwhip.policy.base import OrderingPolicy
from deepbullwhip.registry import register


@register("policy", "smoothing_out")
class SmoothingOUTPolicy(OrderingPolicy):
    """OUT with exponential smoothing on the order quantity.

    O(t) = max(0, alpha_s * (S - IP) + (1 - alpha_s) * O(t-1))

    Smooths the order stream by blending the current OUT decision with
    the previous order. Reduces bullwhip at the cost of slower response
    to demand changes.

    Parameters
    ----------
    lead_time : int
        Replenishment lead time in periods.
    service_level : float
        Target service level (e.g. 0.95).
    alpha_s : float
        Smoothing factor in (0, 1]. Higher values track OUT more closely.
    """

    def __init__(
        self,
        lead_time: int,
        service_level: float = 0.95,
        alpha_s: float = 0.3,
    ) -> None:
        self.lead_time = lead_time
        self.service_level = service_level
        self.z_alpha = stats.norm.ppf(service_level)
        self.alpha_s = alpha_s
        self._prev_order = 0.0

    def compute_order(
        self,
        inventory_position: float,
        forecast_mean: float,
        forecast_std: float,
    ) -> float:
        S = (self.lead_time + 1) * forecast_mean + (
            self.z_alpha * forecast_std * np.sqrt(self.lead_time + 1)
        )
        raw = S - inventory_position
        smoothed = self.alpha_s * raw + (1 - self.alpha_s) * self._prev_order
        order = max(0.0, smoothed)
        self._prev_order = order
        return order

    def reset(self) -> None:
        """Reset internal state for a new simulation run."""
        self._prev_order = 0.0
