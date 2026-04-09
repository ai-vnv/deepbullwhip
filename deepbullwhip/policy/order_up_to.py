import numpy as np
from scipy import stats

from deepbullwhip.policy.base import OrderingPolicy


class OrderUpToPolicy(OrderingPolicy):
    """Order-Up-To (OUT / base-stock) policy.

    S = (L+1) * forecast_mean + z_alpha * forecast_std * sqrt(L+1)
    order = max(0, S - IP)

    Parameters
    ----------
    lead_time : int
        Replenishment lead time in periods.
    service_level : float
        Target service level (e.g. 0.95). Used to set z_alpha.
    """

    def __init__(self, lead_time: int, service_level: float = 0.95) -> None:
        self.lead_time = lead_time
        self.service_level = service_level
        self.z_alpha = stats.norm.ppf(service_level)

    def compute_order(
        self,
        inventory_position: float,
        forecast_mean: float,
        forecast_std: float,
    ) -> float:
        S = (self.lead_time + 1) * forecast_mean + (
            self.z_alpha * forecast_std * np.sqrt(self.lead_time + 1)
        )
        return max(0.0, S - inventory_position)
