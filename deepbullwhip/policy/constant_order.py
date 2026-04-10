"""Constant order policy."""

from deepbullwhip.policy.base import OrderingPolicy
from deepbullwhip.registry import register


@register("policy", "constant_order")
class ConstantOrderPolicy(OrderingPolicy):
    """Order a fixed quantity every period.

    BWR = 0 by construction since Var(orders) = 0.
    Useful as a lower-bound baseline for bullwhip analysis.

    Parameters
    ----------
    order_quantity : float
        Fixed order quantity per period.
    """

    def __init__(self, order_quantity: float = 12.5) -> None:
        self.order_quantity = order_quantity

    def compute_order(
        self,
        inventory_position: float,
        forecast_mean: float,
        forecast_std: float,
    ) -> float:
        return self.order_quantity
