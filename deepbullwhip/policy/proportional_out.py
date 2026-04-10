"""Proportional Order-Up-To (POUT) policy."""

import numpy as np
from scipy import stats

from deepbullwhip.policy.base import OrderingPolicy
from deepbullwhip.registry import register


@register("policy", "proportional_out")
class ProportionalOUTPolicy(OrderingPolicy):
    """POUT: O(t) = alpha * max(0, S(t) - IP(t))

    A smoothed variant of the Order-Up-To policy. The smoothing parameter
    ``alpha`` trades off bullwhip amplification against inventory variance.

    Parameters
    ----------
    lead_time : int
        Replenishment lead time in periods.
    service_level : float
        Target service level (e.g. 0.95).
    alpha : float
        Smoothing factor in (0, 1]. alpha=1 reduces to standard OUT.
        Lower alpha reduces order variance but increases inventory variance.
        See Disney & Towill (2003) for BWR-NSAmp tradeoff analysis.
    """

    def __init__(
        self,
        lead_time: int,
        service_level: float = 0.95,
        alpha: float = 0.5,
    ) -> None:
        self.lead_time = lead_time
        self.service_level = service_level
        self.z_alpha = stats.norm.ppf(service_level)
        self.alpha = alpha

    def compute_order(
        self,
        inventory_position: float,
        forecast_mean: float,
        forecast_std: float,
    ) -> float:
        S = (self.lead_time + 1) * forecast_mean + (
            self.z_alpha * forecast_std * np.sqrt(self.lead_time + 1)
        )
        return max(0.0, self.alpha * (S - inventory_position))
