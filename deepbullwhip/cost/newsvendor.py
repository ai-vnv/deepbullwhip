from deepbullwhip.cost.base import CostFunction


class NewsvendorCost(CostFunction):
    """Newsvendor-style cost: h * inventory^+ + b * inventory^-.

    Parameters
    ----------
    holding_cost : float
        Per-unit per-period holding cost (applied when inventory >= 0).
    backorder_cost : float
        Per-unit per-period backorder cost (applied when inventory < 0).
    """

    def __init__(self, holding_cost: float, backorder_cost: float) -> None:
        self.h = holding_cost
        self.b = backorder_cost

    def compute(self, inventory: float) -> float:
        if inventory >= 0:
            return self.h * inventory
        return self.b * abs(inventory)
