"""Perishable cost function with obsolescence penalty."""

from deepbullwhip.cost.base import CostFunction
from deepbullwhip.registry import register


@register("cost", "perishable")
class PerishableCost(CostFunction):
    """Newsvendor cost + obsolescence penalty for perishable goods.

    C(I) = h * [I]+ + b * [-I]+ + gamma * [I - buffer]+

    The gamma term penalizes inventory above a shelf-life buffer,
    modeling technology perishability in semiconductors.

    Parameters
    ----------
    holding_cost : float
        Per-unit per-period holding cost.
    backorder_cost : float
        Per-unit per-period backorder cost.
    gamma : float
        Obsolescence penalty rate per unit above buffer.
    buffer : float
        Inventory threshold above which obsolescence kicks in.
    """

    def __init__(
        self,
        holding_cost: float,
        backorder_cost: float,
        gamma: float = 0.05,
        buffer: float = 50.0,
    ) -> None:
        self.h = holding_cost
        self.b = backorder_cost
        self.gamma = gamma
        self.buffer = buffer

    def compute(self, inventory: float) -> float:
        base = self.h * max(inventory, 0) + self.b * max(-inventory, 0)
        obsolescence = self.gamma * max(0, inventory - self.buffer)
        return base + obsolescence
