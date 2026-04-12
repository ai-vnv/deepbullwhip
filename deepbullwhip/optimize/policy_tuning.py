"""Simulation-based policy parameter tuning using Pyomo.

Uses a simulation-optimization hybrid approach: Pyomo manages the
outer optimization loop (searching over policy parameters), while
DeepBullwhip's simulation engine evaluates each candidate solution.

This avoids the need to reformulate complex simulation dynamics as
algebraic constraints.

Requires the ``pyomo`` optional dependency::

    pip install deepbullwhip[optimize]

Functions
---------
tune_service_levels
    Find optimal per-echelon service levels.
tune_smoothing_factors
    Find optimal smoothing factors for SmoothingOUT policies.

Examples
--------
>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.chain.graph import from_serial
>>> from deepbullwhip.optimize.policy_tuning import tune_service_levels
>>> import numpy as np
>>>
>>> graph = from_serial(beer_game_config())
>>> scenarios = np.random.default_rng(42).normal(10, 2, (20, 52))
>>> result = tune_service_levels(graph, np.maximum(scenarios, 0))  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from deepbullwhip._optional import import_optional
from deepbullwhip._types import TimeSeries
from deepbullwhip.chain.graph import SupplyChainGraph


@dataclass
class PolicyTuningResult:
    """Results from policy parameter tuning.

    Parameters
    ----------
    parameters : dict[str, float]
        Optimal parameter value for each node.
    objective_value : float
        Best objective function value achieved.
    n_evaluations : int
        Number of simulation evaluations performed.
    """

    parameters: dict[str, float]
    objective_value: float
    n_evaluations: int


def _simulate_with_service_levels(
    graph: SupplyChainGraph,
    service_levels: dict[str, float],
    demand_scenarios: np.ndarray,
    objective: str,
) -> float:
    """Run simulation with given service levels and return objective value."""
    from deepbullwhip.chain.network_sim import NetworkSupplyChain
    from deepbullwhip.policy.order_up_to import OrderUpToPolicy

    policies = {
        name: OrderUpToPolicy(
            lead_time=cfg.lead_time,
            service_level=service_levels[name],
        )
        for name, cfg in graph.nodes.items()
    }

    chain = NetworkSupplyChain(graph, policies=policies)
    demand_nodes = graph.demand_nodes

    total_obj = 0.0
    n_scenarios = demand_scenarios.shape[0]

    for s in range(n_scenarios):
        scenario = demand_scenarios[s]
        demand = {dn: scenario for dn in demand_nodes}
        fm = {dn: np.full_like(scenario, scenario.mean()) for dn in demand_nodes}
        fs = {dn: np.full_like(scenario, scenario.std()) for dn in demand_nodes}

        result = chain.simulate(demand, fm, fs)

        if objective == "total_cost":
            total_obj += result.total_cost
        elif objective == "bullwhip":
            total_obj += result.cumulative_bullwhip
        elif objective == "weighted":
            total_obj += result.total_cost + 100.0 * result.cumulative_bullwhip
        else:
            raise ValueError(f"Unknown objective: {objective!r}")

    return total_obj / n_scenarios


def tune_service_levels(
    graph: SupplyChainGraph,
    demand_scenarios: np.ndarray,
    objective: str = "total_cost",
    grid_points: int = 10,
    bounds: tuple[float, float] = (0.80, 0.99),
) -> PolicyTuningResult:
    """Find optimal service level for each echelon.

    Uses grid search over service level values, evaluating each
    combination via simulation. For small networks, this is exhaustive;
    for larger networks, coordinate-descent is used.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology and node configurations.
    demand_scenarios : numpy.ndarray
        Demand scenarios, shape ``(N, T)``.
    objective : str
        Optimization objective:

        - ``"total_cost"``: minimize expected total cost
        - ``"bullwhip"``: minimize cumulative bullwhip ratio
        - ``"weighted"``: minimize cost + 100 * bullwhip
    grid_points : int
        Number of grid points per dimension (default 10).
    bounds : tuple[float, float]
        Service level search range (default ``(0.80, 0.99)``).

    Returns
    -------
    PolicyTuningResult
        Optimal service levels and objective value.

    Examples
    --------
    >>> import numpy as np
    >>> from deepbullwhip.chain.config import consumer_2tier_config
    >>> from deepbullwhip.chain.graph import from_serial
    >>> graph = from_serial(consumer_2tier_config())
    >>> scenarios = np.random.default_rng(0).normal(10, 2, (10, 26))
    >>> result = tune_service_levels(graph, np.maximum(scenarios, 0))
    >>> isinstance(result.parameters, dict)
    True
    """
    if demand_scenarios.ndim == 1:
        demand_scenarios = demand_scenarios.reshape(1, -1)

    node_names = list(graph.nodes.keys())
    grid = np.linspace(bounds[0], bounds[1], grid_points)

    n_evaluations = 0

    if len(node_names) <= 3:
        # Exhaustive grid search for small networks
        best_obj = float("inf")
        best_params: dict[str, float] = {n: 0.95 for n in node_names}

        # Build grid combinations
        from itertools import product

        for combo in product(grid, repeat=len(node_names)):
            svc_levels = dict(zip(node_names, combo))
            obj = _simulate_with_service_levels(
                graph, svc_levels, demand_scenarios, objective
            )
            n_evaluations += 1

            if obj < best_obj:
                best_obj = obj
                best_params = svc_levels.copy()
    else:
        # Coordinate descent for larger networks
        best_params = {n: graph.nodes[n].service_level for n in node_names}
        best_obj = _simulate_with_service_levels(
            graph, best_params, demand_scenarios, objective
        )
        n_evaluations += 1

        for _iteration in range(3):  # 3 sweeps
            for name in node_names:
                local_best_obj = best_obj
                local_best_val = best_params[name]

                for val in grid:
                    candidate = best_params.copy()
                    candidate[name] = float(val)
                    obj = _simulate_with_service_levels(
                        graph, candidate, demand_scenarios, objective
                    )
                    n_evaluations += 1

                    if obj < local_best_obj:
                        local_best_obj = obj
                        local_best_val = float(val)

                best_params[name] = local_best_val
                best_obj = local_best_obj

    return PolicyTuningResult(
        parameters=best_params,
        objective_value=best_obj,
        n_evaluations=n_evaluations,
    )


def _simulate_with_smoothing(
    graph: SupplyChainGraph,
    alphas: dict[str, float],
    demand_scenarios: np.ndarray,
) -> float:
    """Run simulation with SmoothingOUT policies and return total cost."""
    from deepbullwhip.chain.network_sim import NetworkSupplyChain
    from deepbullwhip.policy.smoothing_out import SmoothingOUTPolicy

    policies = {
        name: SmoothingOUTPolicy(
            lead_time=cfg.lead_time,
            service_level=cfg.service_level,
            alpha_s=alphas[name],
        )
        for name, cfg in graph.nodes.items()
    }

    chain = NetworkSupplyChain(graph, policies=policies)
    demand_nodes = graph.demand_nodes

    total_cost = 0.0
    n_scenarios = demand_scenarios.shape[0]

    for s in range(n_scenarios):
        scenario = demand_scenarios[s]
        demand = {dn: scenario for dn in demand_nodes}
        fm = {dn: np.full_like(scenario, scenario.mean()) for dn in demand_nodes}
        fs = {dn: np.full_like(scenario, scenario.std()) for dn in demand_nodes}

        result = chain.simulate(demand, fm, fs)
        total_cost += result.total_cost

    return total_cost / n_scenarios


def tune_smoothing_factors(
    graph: SupplyChainGraph,
    demand_scenarios: np.ndarray,
    grid_points: int = 10,
    bounds: tuple[float, float] = (0.1, 1.0),
) -> PolicyTuningResult:
    """Find optimal smoothing factors for SmoothingOUT policies.

    Uses coordinate descent over smoothing factor ``alpha_s`` for each
    node, minimizing expected total cost via simulation evaluation.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology and node configurations.
    demand_scenarios : numpy.ndarray
        Demand scenarios, shape ``(N, T)``.
    grid_points : int
        Number of grid points per dimension (default 10).
    bounds : tuple[float, float]
        Smoothing factor search range (default ``(0.1, 1.0)``).

    Returns
    -------
    PolicyTuningResult
        Optimal smoothing factors and total cost.

    Examples
    --------
    >>> import numpy as np
    >>> from deepbullwhip.chain.config import consumer_2tier_config
    >>> from deepbullwhip.chain.graph import from_serial
    >>> graph = from_serial(consumer_2tier_config())
    >>> scenarios = np.random.default_rng(0).normal(10, 2, (10, 26))
    >>> result = tune_smoothing_factors(graph, np.maximum(scenarios, 0))
    >>> all(0.1 <= v <= 1.0 for v in result.parameters.values())
    True
    """
    if demand_scenarios.ndim == 1:
        demand_scenarios = demand_scenarios.reshape(1, -1)

    node_names = list(graph.nodes.keys())
    grid = np.linspace(bounds[0], bounds[1], grid_points)

    # Coordinate descent
    best_params = {n: 0.5 for n in node_names}
    best_obj = _simulate_with_smoothing(graph, best_params, demand_scenarios)
    n_evaluations = 1

    for _iteration in range(3):  # 3 sweeps
        for name in node_names:
            local_best_obj = best_obj
            local_best_val = best_params[name]

            for val in grid:
                candidate = best_params.copy()
                candidate[name] = float(val)
                obj = _simulate_with_smoothing(
                    graph, candidate, demand_scenarios
                )
                n_evaluations += 1

                if obj < local_best_obj:
                    local_best_obj = obj
                    local_best_val = float(val)

            best_params[name] = local_best_val
            best_obj = local_best_obj

    return PolicyTuningResult(
        parameters=best_params,
        objective_value=best_obj,
        n_evaluations=n_evaluations,
    )
