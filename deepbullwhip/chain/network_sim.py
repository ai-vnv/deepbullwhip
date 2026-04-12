"""Network supply chain simulator for arbitrary DAG topologies.

Generalises :class:`~deepbullwhip.chain.serial.SerialSupplyChain` from
serial (linear) chains to directed acyclic graphs with branching and
merging material flows.

Examples
--------
>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.chain.graph import from_serial
>>> from deepbullwhip.chain.network_sim import NetworkSupplyChain
>>> import numpy as np
>>>
>>> graph = from_serial(beer_game_config())
>>> chain = NetworkSupplyChain(graph)
>>> demand = np.full(52, 4.0)
>>> result = chain.simulate(
...     demand={"Retailer": demand},
...     forecasts_mean={"Retailer": np.full(52, 4.0)},
...     forecasts_std={"Retailer": np.full(52, 1.0)},
... )
>>> result.node_results["Retailer"].fill_rate > 0.8
True
"""

from __future__ import annotations

from typing import Any

import numpy as np

from deepbullwhip._types import (
    EchelonResult,
    NetworkSimulationResult,
    TimeSeries,
)
from deepbullwhip.chain.config import EchelonConfig
from deepbullwhip.chain.echelon import SupplyChainEchelon
from deepbullwhip.chain.graph import SupplyChainGraph, from_serial
from deepbullwhip.cost.newsvendor import NewsvendorCost
from deepbullwhip.policy.order_up_to import OrderUpToPolicy


class NetworkSupplyChain:
    """Supply chain simulator for arbitrary DAG topologies.

    Simulates material flow through a directed acyclic graph where:

    - **Demand-facing nodes** (no outgoing edges) receive external demand.
    - **Source nodes** (no incoming edges) have unlimited raw material.
    - **Interior nodes** face demand equal to the sum of orders from
      all downstream neighbors.

    Processing follows reverse topological order (demand-facing first,
    then progressively upstream), matching the information flow in
    real supply chains.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology and node configurations.
    policies : dict[str, OrderingPolicy] or None
        Per-node ordering policies. If ``None``, each node uses an
        :class:`~deepbullwhip.policy.order_up_to.OrderUpToPolicy`
        with the node's configured lead time and service level.
    cost_fns : dict[str, CostFunction] or None
        Per-node cost functions. If ``None``, each node uses a
        :class:`~deepbullwhip.cost.newsvendor.NewsvendorCost` with
        the node's configured holding and backorder costs.

    Examples
    --------
    Build a distribution tree with two retailers sharing a warehouse:

    >>> from deepbullwhip.chain.config import EchelonConfig
    >>> from deepbullwhip.chain.graph import SupplyChainGraph, EdgeConfig
    >>> graph = SupplyChainGraph(
    ...     nodes={
    ...         "Factory": EchelonConfig("Factory", 4, 0.10, 0.40),
    ...         "Warehouse": EchelonConfig("Warehouse", 2, 0.15, 0.50),
    ...         "Retail_A": EchelonConfig("Retail_A", 1, 0.20, 0.60),
    ...         "Retail_B": EchelonConfig("Retail_B", 1, 0.20, 0.60),
    ...     },
    ...     edges={
    ...         ("Factory", "Warehouse"): EdgeConfig(lead_time=3),
    ...         ("Warehouse", "Retail_A"): EdgeConfig(lead_time=1),
    ...         ("Warehouse", "Retail_B"): EdgeConfig(lead_time=1),
    ...     },
    ... )
    >>> chain = NetworkSupplyChain(graph)
    """

    def __init__(
        self,
        graph: SupplyChainGraph,
        policies: dict[str, Any] | None = None,
        cost_fns: dict[str, Any] | None = None,
    ) -> None:
        self.graph = graph
        self._topo_order = graph.topological_order()

        # Build echelon objects for each node
        self._echelons: dict[str, SupplyChainEchelon] = {}
        for name, cfg in graph.nodes.items():
            policy = (
                policies[name]
                if policies and name in policies
                else OrderUpToPolicy(
                    lead_time=cfg.lead_time, service_level=cfg.service_level
                )
            )
            total_h = cfg.holding_cost + cfg.depreciation_rate
            cost_fn = (
                cost_fns[name]
                if cost_fns and name in cost_fns
                else NewsvendorCost(
                    holding_cost=total_h, backorder_cost=cfg.backorder_cost
                )
            )
            self._echelons[name] = SupplyChainEchelon(
                name=name,
                lead_time=cfg.lead_time,
                policy=policy,
                cost_fn=cost_fn,
                initial_inventory=cfg.initial_inventory,
            )

    @classmethod
    def from_serial(
        cls,
        configs: list[EchelonConfig],
        **kwargs: Any,
    ) -> NetworkSupplyChain:
        """Build from a serial chain config list.

        Convenience constructor for backward compatibility with
        :class:`~deepbullwhip.chain.serial.SerialSupplyChain`.

        Parameters
        ----------
        configs : list[EchelonConfig]
            Echelon configs ordered from downstream to upstream.
        **kwargs
            Forwarded to :class:`NetworkSupplyChain`.

        Returns
        -------
        NetworkSupplyChain
        """
        return cls(from_serial(configs), **kwargs)

    @classmethod
    def from_networkx(cls, G: Any, **kwargs: Any) -> NetworkSupplyChain:
        """Build from a NetworkX ``DiGraph``.

        Parameters
        ----------
        G : nx.DiGraph
            A directed acyclic graph with node/edge attributes.
        **kwargs
            Forwarded to :class:`NetworkSupplyChain`.

        Returns
        -------
        NetworkSupplyChain

        Raises
        ------
        ImportError
            If ``networkx`` is not installed.
        """
        from deepbullwhip.network.convert import from_networkx

        return cls(from_networkx(G), **kwargs)

    def reset(self) -> None:
        """Reset all echelons to initial conditions."""
        for echelon in self._echelons.values():
            echelon.reset()

    def simulate(
        self,
        demand: dict[str, TimeSeries],
        forecasts_mean: dict[str, TimeSeries],
        forecasts_std: dict[str, TimeSeries],
    ) -> NetworkSimulationResult:
        """Run the full simulation over T periods.

        Parameters
        ----------
        demand : dict[str, TimeSeries]
            External demand time series for each demand-facing node.
            Keys must include all nodes returned by
            :attr:`SupplyChainGraph.demand_nodes`.
        forecasts_mean : dict[str, TimeSeries]
            Point forecasts for demand-facing nodes.
        forecasts_std : dict[str, TimeSeries]
            Forecast error std for demand-facing nodes.

        Returns
        -------
        NetworkSimulationResult
            Per-node results and per-edge material flows.

        Raises
        ------
        ValueError
            If demand is not provided for all demand-facing nodes.

        Examples
        --------
        >>> import numpy as np
        >>> from deepbullwhip.chain.config import beer_game_config
        >>> from deepbullwhip.chain.graph import from_serial
        >>> chain = NetworkSupplyChain(from_serial(beer_game_config()))
        >>> T = 52
        >>> d = np.full(T, 4.0)
        >>> result = chain.simulate(
        ...     demand={"Retailer": d},
        ...     forecasts_mean={"Retailer": np.full(T, 4.0)},
        ...     forecasts_std={"Retailer": np.full(T, 1.0)},
        ... )
        """
        # Validate demand nodes
        expected_demand_nodes = set(self.graph.demand_nodes)
        provided_demand_nodes = set(demand.keys())
        missing = expected_demand_nodes - provided_demand_nodes
        if missing:
            raise ValueError(
                f"Missing demand for demand-facing nodes: {missing}. "
                f"Expected: {expected_demand_nodes}"
            )

        # Determine simulation length from first demand series
        first_demand = next(iter(demand.values()))
        T = len(first_demand)

        self.reset()

        # Track orders per node per period for upstream demand propagation
        node_orders: dict[str, list[float]] = {n: [] for n in self.graph.nodes}
        # Track edge flows
        edge_flows: dict[tuple[str, str], list[float]] = {
            e: [] for e in self.graph.edges
        }

        # Process in reverse topological order (demand-facing first)
        sim_order = list(reversed(self._topo_order))

        for t in range(T):
            period_orders: dict[str, float] = {}

            for node_name in sim_order:
                echelon = self._echelons[node_name]
                downstream = self.graph.downstream_neighbors(node_name)

                if not downstream:
                    # Demand-facing node: use external demand
                    d = float(demand[node_name][t])
                    fm = float(forecasts_mean[node_name][t])
                    fs = float(forecasts_std[node_name][t])
                else:
                    # Interior/source node: demand = sum of downstream orders
                    d = sum(period_orders[dn] for dn in downstream)

                    # Forecast from recent downstream orders
                    all_downstream_orders: list[float] = []
                    for dn in downstream:
                        all_downstream_orders.extend(node_orders[dn])

                    if all_downstream_orders:
                        recent = all_downstream_orders[-min(8, len(all_downstream_orders)):]
                        fm = float(np.mean(recent))
                        fs = float(np.std(recent)) if len(recent) > 1 else 1.0
                    else:
                        # First period fallback: use first available forecast
                        any_demand_node = next(iter(forecasts_mean))
                        fm = float(forecasts_mean[any_demand_node][t])
                        fs = float(forecasts_std[any_demand_node][t])

                order = echelon.step(d, fm, fs)
                period_orders[node_name] = order
                node_orders[node_name].append(order)

                # Record edge flows (order flows upstream along incoming edges)
                for dn in downstream:
                    edge = (node_name, dn)
                    if edge in edge_flows:
                        edge_flows[edge].append(period_orders[dn])

        return self._compute_results(demand, edge_flows)

    def _compute_results(
        self,
        demand: dict[str, TimeSeries],
        edge_flows: dict[tuple[str, str], list[float]],
    ) -> NetworkSimulationResult:
        """Compute per-node metrics and build the result object."""
        node_results: dict[str, EchelonResult] = {}

        # Compute variance of external demand (for bullwhip denominators)
        all_demand = np.concatenate(list(demand.values()))
        var_demand = float(np.var(all_demand)) if len(all_demand) > 0 else 1.0

        for name, echelon in self._echelons.items():
            orders = echelon.orders_array
            inv = echelon.inventory_array
            costs = echelon.costs_array

            var_orders = float(np.var(orders)) if len(orders) > 0 else 0.0

            # Bullwhip ratio: Var(orders) / Var(incoming demand)
            upstream = self.graph.upstream_neighbors(name)
            if not upstream:
                # Source node: compare to demand from downstream
                downstream = self.graph.downstream_neighbors(name)
                if downstream:
                    downstream_orders = np.concatenate([
                        self._echelons[dn].orders_array for dn in downstream
                    ])
                    var_incoming = float(np.var(downstream_orders))
                else:
                    var_incoming = var_demand
            else:
                var_incoming = var_demand

            bw = var_orders / var_incoming if var_incoming > 0 else 1.0

            node_results[name] = EchelonResult(
                name=name,
                orders=orders,
                inventory_levels=inv,
                costs=costs,
                bullwhip_ratio=bw,
                fill_rate=float(np.mean(inv >= 0)),
                total_cost=float(costs.sum()),
            )

        # Edge flows as numpy arrays
        edge_flows_np: dict[tuple[str, str], TimeSeries] = {
            edge: np.asarray(flows, dtype=np.float64)
            for edge, flows in edge_flows.items()
        }

        # Cumulative bullwhip: max across source nodes
        source_nodes = self.graph.source_nodes
        if source_nodes:
            cum_bw = max(
                node_results[n].bullwhip_ratio for n in source_nodes
            )
        else:
            cum_bw = 1.0

        total_cost = sum(er.total_cost for er in node_results.values())

        return NetworkSimulationResult(
            node_results=node_results,
            edge_flows=edge_flows_np,
            cumulative_bullwhip=cum_bw,
            total_cost=total_cost,
        )
