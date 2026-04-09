from __future__ import annotations

import numpy as np

from deepbullwhip._types import EchelonResult, SimulationResult, TimeSeries
from deepbullwhip.chain.config import EchelonConfig, default_semiconductor_config
from deepbullwhip.chain.echelon import SupplyChainEchelon
from deepbullwhip.cost.newsvendor import NewsvendorCost
from deepbullwhip.policy.order_up_to import OrderUpToPolicy


class SerialSupplyChain:
    """K-echelon serial supply chain.

    Parameters
    ----------
    echelons : list[SupplyChainEchelon] or None
        Pre-built echelons. If None, uses default_semiconductor_config().
    """

    def __init__(self, echelons: list[SupplyChainEchelon] | None = None) -> None:
        if echelons is None:
            echelons = self._from_config(default_semiconductor_config())
        self.echelons = echelons
        self.K = len(echelons)

    @staticmethod
    def _from_config(configs: list[EchelonConfig]) -> list[SupplyChainEchelon]:
        result = []
        for cfg in configs:
            total_h = cfg.holding_cost + cfg.depreciation_rate
            policy = OrderUpToPolicy(
                lead_time=cfg.lead_time, service_level=cfg.service_level
            )
            cost_fn = NewsvendorCost(
                holding_cost=total_h, backorder_cost=cfg.backorder_cost
            )
            result.append(
                SupplyChainEchelon(
                    name=cfg.name,
                    lead_time=cfg.lead_time,
                    policy=policy,
                    cost_fn=cost_fn,
                    initial_inventory=cfg.initial_inventory,
                )
            )
        return result

    @classmethod
    def from_config(cls, configs: list[EchelonConfig]) -> SerialSupplyChain:
        """Build a chain from a list of EchelonConfig objects."""
        return cls(echelons=cls._from_config(configs))

    def reset(self) -> None:
        for e in self.echelons:
            e.reset()

    def simulate(
        self,
        demand: TimeSeries,
        forecasts_mean: TimeSeries,
        forecasts_std: TimeSeries,
    ) -> SimulationResult:
        """Run the full simulation over T periods.

        Parameters
        ----------
        demand : array, shape (T,)
        forecasts_mean : array, shape (T,)
            Point forecasts for echelon 1 (end-customer demand).
        forecasts_std : array, shape (T,)
            Forecast error std estimates for echelon 1.

        Returns
        -------
        SimulationResult
        """
        T = len(demand)
        self.reset()

        for t in range(T):
            d = demand[t]
            fm = forecasts_mean[t]
            fs = forecasts_std[t]

            # Echelon 1 faces end-customer demand
            order = self.echelons[0].step(d, fm, fs)

            # Each subsequent echelon faces orders from the previous
            for k in range(1, self.K):
                upstream_demand = order
                if t > 0:
                    recent = self.echelons[k - 1].orders[
                        -min(8, len(self.echelons[k - 1].orders)) :
                    ]
                    upstream_fm = float(np.mean(recent))
                    upstream_fs = (
                        float(np.std(recent)) if len(recent) > 1 else fs
                    )
                else:
                    upstream_fm = fm
                    upstream_fs = fs
                order = self.echelons[k].step(upstream_demand, upstream_fm, upstream_fs)

        return self._compute_results(demand)

    def _compute_results(self, demand: TimeSeries) -> SimulationResult:
        var_demand = float(np.var(demand))
        echelon_results: list[EchelonResult] = []

        for k, e in enumerate(self.echelons):
            orders = e.orders_array
            inv = e.inventory_array
            costs = e.costs_array

            if k == 0:
                bw = float(np.var(orders) / var_demand) if var_demand > 0 else 1.0
            else:
                var_prev = float(np.var(self.echelons[k - 1].orders_array))
                bw = float(np.var(orders) / var_prev) if var_prev > 0 else 1.0

            echelon_results.append(
                EchelonResult(
                    name=e.name,
                    orders=orders,
                    inventory_levels=inv,
                    costs=costs,
                    bullwhip_ratio=bw,
                    fill_rate=float(np.mean(inv >= 0)),
                    total_cost=float(costs.sum()),
                )
            )

        all_orders_K = self.echelons[-1].orders_array
        cum_bw = float(np.var(all_orders_K) / var_demand) if var_demand > 0 else 1.0
        total_cost = sum(er.total_cost for er in echelon_results)

        return SimulationResult(
            echelon_results=echelon_results,
            cumulative_bullwhip=cum_bw,
            total_cost=total_cost,
        )
