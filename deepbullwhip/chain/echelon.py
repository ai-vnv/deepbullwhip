from __future__ import annotations

import numpy as np

from deepbullwhip._types import TimeSeries
from deepbullwhip.cost.base import CostFunction
from deepbullwhip.policy.base import OrderingPolicy


class SupplyChainEchelon:
    """Single echelon in a serial supply chain.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. "Distributor").
    lead_time : int
        Replenishment lead time in periods.
    policy : OrderingPolicy
        The ordering policy (must already know its own lead_time).
    cost_fn : CostFunction
        The per-period cost function.
    initial_inventory : float
        Starting on-hand inventory.
    """

    def __init__(
        self,
        name: str,
        lead_time: int,
        policy: OrderingPolicy,
        cost_fn: CostFunction,
        initial_inventory: float = 50.0,
    ) -> None:
        self.name = name
        self.lead_time = lead_time
        self.policy = policy
        self.cost_fn = cost_fn
        self.initial_inventory = initial_inventory

        self.inventory: float = 0.0
        self.pipeline: list[float] = []
        self.orders: list[float] = []
        self.inventory_levels: list[float] = []
        self.costs: list[float] = []

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.inventory = self.initial_inventory
        self.pipeline = [0.0] * self.lead_time
        self.orders = []
        self.inventory_levels = []
        self.costs = []

    def step(self, demand: float, forecast_mean: float, forecast_std: float) -> float:
        """Execute one period: receive, order, satisfy demand, compute cost.

        Returns the order quantity placed this period.
        """
        # Receive oldest pipeline order
        if self.pipeline:
            self.inventory += self.pipeline.pop(0)

        # Inventory position = on-hand + pipeline
        ip = self.inventory + sum(self.pipeline)

        # Ordering decision (delegated to policy)
        order = self.policy.compute_order(ip, forecast_mean, forecast_std)
        self.orders.append(order)
        self.pipeline.append(order)

        # Satisfy demand
        self.inventory -= demand

        # Cost (delegated to cost function)
        cost = self.cost_fn.compute(self.inventory)
        self.inventory_levels.append(self.inventory)
        self.costs.append(cost)

        return order

    @property
    def orders_array(self) -> TimeSeries:
        return np.asarray(self.orders, dtype=np.float64)

    @property
    def inventory_array(self) -> TimeSeries:
        return np.asarray(self.inventory_levels, dtype=np.float64)

    @property
    def costs_array(self) -> TimeSeries:
        return np.asarray(self.costs, dtype=np.float64)
