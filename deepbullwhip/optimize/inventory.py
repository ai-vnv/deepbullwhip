"""Multi-echelon inventory optimization using Pyomo.

Formulates a scenario-based stochastic programming model to find
optimal base-stock levels that minimize expected total cost subject
to service level constraints.

The model uses demand scenarios (from Monte Carlo simulation or
historical data) to approximate the stochastic demand distribution.

Requires the ``pyomo`` optional dependency and a compatible solver
(e.g., ``glpk``, ``cbc``, ``gurobi``)::

    pip install deepbullwhip[optimize]

Functions
---------
build_inventory_model
    Construct a Pyomo ``ConcreteModel`` for inventory optimization.
solve_model
    Solve the model and extract optimal base-stock levels.

Examples
--------
>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.chain.graph import from_serial
>>> from deepbullwhip.optimize.inventory import build_inventory_model, solve_model
>>> import numpy as np
>>>
>>> graph = from_serial(beer_game_config())
>>> # Generate demand scenarios: 50 paths of 52 periods
>>> rng = np.random.default_rng(42)
>>> scenarios = rng.normal(loc=10.0, scale=2.0, size=(50, 52))
>>> scenarios = np.maximum(scenarios, 0)
>>>
>>> model = build_inventory_model(graph, scenarios)
>>> result = solve_model(model, solver="glpk")  # doctest: +SKIP
>>> print(result)  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from deepbullwhip._optional import import_optional
from deepbullwhip.chain.graph import SupplyChainGraph


@dataclass
class InventoryOptResult:
    """Results from inventory optimization.

    Parameters
    ----------
    base_stock_levels : dict[str, float]
        Optimal base-stock level for each node.
    expected_cost : float
        Expected total cost under optimal policy.
    solver_status : str
        Solver termination condition.
    """

    base_stock_levels: dict[str, float]
    expected_cost: float
    solver_status: str


def build_inventory_model(
    graph: SupplyChainGraph,
    demand_scenarios: np.ndarray,
    service_levels: dict[str, float] | None = None,
) -> Any:
    """Build a Pyomo model for multi-echelon inventory optimization.

    Formulates a linear program to find base-stock levels that
    minimize expected total holding and backorder costs across all
    scenarios, subject to per-node service level constraints.

    **Decision variables**: Base-stock level :math:`S_i` for each node
    :math:`i`.

    **Objective**: Minimize

    .. math::

        \\frac{1}{N} \\sum_{s=1}^{N} \\sum_{i} \\sum_{t=1}^{T}
        \\left[ h_i \\cdot I_{i,t,s}^+ + b_i \\cdot I_{i,t,s}^- \\right]

    **Constraints**: For each node :math:`i`:

    .. math::

        \\frac{1}{NT} \\sum_{s,t} \\mathbf{1}[I_{i,t,s} \\geq 0]
        \\geq \\alpha_i

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology and node configurations.
    demand_scenarios : numpy.ndarray
        Demand scenarios, shape ``(N, T)`` where ``N`` is the number
        of scenarios and ``T`` the number of periods. Applied to
        demand-facing nodes.
    service_levels : dict[str, float] or None
        Per-node minimum service level (fill rate). Defaults to each
        node's configured ``service_level``.

    Returns
    -------
    pyomo.ConcreteModel
        A Pyomo model ready to be solved with :func:`solve_model`.

    Raises
    ------
    ImportError
        If ``pyomo`` is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from deepbullwhip.chain.config import consumer_2tier_config
    >>> from deepbullwhip.chain.graph import from_serial
    >>> graph = from_serial(consumer_2tier_config())
    >>> scenarios = np.random.default_rng(0).normal(10, 2, (20, 26))
    >>> model = build_inventory_model(graph, np.maximum(scenarios, 0))
    """
    import_optional("pyomo", "optimize")
    from pyomo.environ import (
        ConcreteModel,
        NonNegativeReals,
        Objective,
        Param,
        RangeSet,
        Set,
        Var,
        minimize,
    )

    if demand_scenarios.ndim == 1:
        demand_scenarios = demand_scenarios.reshape(1, -1)

    N, T = demand_scenarios.shape
    node_names = list(graph.nodes.keys())

    # Resolve service levels
    svc = {}
    for name, cfg in graph.nodes.items():
        if service_levels and name in service_levels:
            svc[name] = service_levels[name]
        else:
            svc[name] = cfg.service_level

    model = ConcreteModel("InventoryOptimization")

    # Sets
    model.Nodes = Set(initialize=node_names)
    model.Scenarios = RangeSet(0, N - 1)
    model.Periods = RangeSet(0, T - 1)

    # Parameters
    model.holding_cost = Param(
        model.Nodes,
        initialize={n: cfg.holding_cost + cfg.depreciation_rate
                    for n, cfg in graph.nodes.items()},
    )
    model.backorder_cost = Param(
        model.Nodes,
        initialize={n: cfg.backorder_cost for n, cfg in graph.nodes.items()},
    )
    model.service_level = Param(
        model.Nodes,
        initialize=svc,
    )
    model.lead_time = Param(
        model.Nodes,
        initialize={n: cfg.lead_time for n, cfg in graph.nodes.items()},
    )

    demand_dict = {}
    for s in range(N):
        for t in range(T):
            demand_dict[s, t] = float(demand_scenarios[s, t])
    model.demand = Param(
        model.Scenarios, model.Periods,
        initialize=demand_dict,
    )

    # Decision variables: base-stock levels
    model.S = Var(model.Nodes, domain=NonNegativeReals)

    # Auxiliary variables for inventory decomposition
    model.inv_plus = Var(
        model.Nodes, model.Scenarios, model.Periods,
        domain=NonNegativeReals,
    )
    model.inv_minus = Var(
        model.Nodes, model.Scenarios, model.Periods,
        domain=NonNegativeReals,
    )

    # Objective: minimize expected total cost
    def cost_rule(m):
        return (1.0 / N) * sum(
            m.holding_cost[n] * m.inv_plus[n, s, t]
            + m.backorder_cost[n] * m.inv_minus[n, s, t]
            for n in m.Nodes
            for s in m.Scenarios
            for t in m.Periods
        )

    model.total_cost = Objective(rule=cost_rule, sense=minimize)

    # Constraints: inventory balance
    # Simplified: inv = S - cumulative_demand_over_lead_time
    from pyomo.environ import Constraint

    def inventory_balance_rule(m, n, s, t):
        lt = int(m.lead_time[n])
        # Demand faced by this node over lead time + review period
        start = max(0, t - lt)
        cum_demand = sum(m.demand[s, tau] for tau in range(start, t + 1))
        # inv_plus - inv_minus = S - cumulative demand
        return m.inv_plus[n, s, t] - m.inv_minus[n, s, t] == m.S[n] - cum_demand

    model.inv_balance = Constraint(
        model.Nodes, model.Scenarios, model.Periods,
        rule=inventory_balance_rule,
    )

    # Service level constraints (linearized)
    # Average fill rate >= target
    # We use a big-M formulation to linearize the indicator
    from pyomo.environ import Binary

    model.stockout = Var(
        model.Nodes, model.Scenarios, model.Periods,
        domain=Binary,
    )

    M = float(demand_scenarios.max() * T * 2)

    def stockout_link_rule(m, n, s, t):
        return m.inv_minus[n, s, t] <= M * m.stockout[n, s, t]

    model.stockout_link = Constraint(
        model.Nodes, model.Scenarios, model.Periods,
        rule=stockout_link_rule,
    )

    def service_level_rule(m, n):
        total_periods = N * T
        return (
            sum(1 - m.stockout[n, s, t]
                for s in m.Scenarios for t in m.Periods)
            >= m.service_level[n] * total_periods
        )

    model.svc_constraint = Constraint(
        model.Nodes,
        rule=service_level_rule,
    )

    # Store graph reference for result extraction
    model._graph = graph

    return model


def solve_model(
    model: Any,
    solver: str = "glpk",
    time_limit: int | None = None,
    tee: bool = False,
) -> InventoryOptResult:
    """Solve the inventory optimization model.

    Parameters
    ----------
    model : pyomo.ConcreteModel
        A model built with :func:`build_inventory_model`.
    solver : str
        Solver name. Common choices: ``"glpk"`` (open source),
        ``"cbc"`` (open source), ``"gurobi"`` (commercial).
    time_limit : int or None
        Maximum solve time in seconds.
    tee : bool
        If ``True``, print solver output to stdout.

    Returns
    -------
    InventoryOptResult
        Optimal base-stock levels and expected cost.

    Raises
    ------
    ImportError
        If ``pyomo`` is not installed.
    RuntimeError
        If the solver fails or no feasible solution is found.

    Examples
    --------
    >>> result = solve_model(model, solver="glpk")  # doctest: +SKIP
    >>> result.base_stock_levels  # doctest: +SKIP
    {'Retailer': 25.3, 'Manufacturer': 42.1}
    """
    import_optional("pyomo", "optimize")
    from pyomo.environ import SolverFactory, value
    from pyomo.opt import TerminationCondition

    opt = SolverFactory(solver)
    if time_limit is not None:
        if solver == "glpk":
            opt.options["tmlim"] = time_limit
        elif solver in ("cbc", "gurobi"):
            opt.options["seconds"] = time_limit

    results = opt.solve(model, tee=tee)

    status = str(results.solver.termination_condition)
    if results.solver.termination_condition not in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    ):
        raise RuntimeError(
            f"Solver did not find a feasible solution. "
            f"Termination condition: {status}"
        )

    # Extract base-stock levels
    base_stock = {}
    for n in model.Nodes:
        base_stock[n] = float(value(model.S[n]))

    expected_cost = float(value(model.total_cost))

    return InventoryOptResult(
        base_stock_levels=base_stock,
        expected_cost=expected_cost,
        solver_status=status,
    )
