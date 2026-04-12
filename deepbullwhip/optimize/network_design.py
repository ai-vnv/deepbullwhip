"""Facility location and network design optimization (experimental).

Formulates a mixed-integer program (MIP) for supply chain network
design: which facilities to open, what capacity to assign, and how
to route material flows to minimize total cost.

.. warning::

    This module is **experimental** and the API may change in future
    releases. The current formulation is a single-period, deterministic
    facility location model suitable for strategic planning.

Requires the ``pyomo`` optional dependency and a MIP-capable solver
(e.g., ``glpk``, ``cbc``, ``gurobi``)::

    pip install deepbullwhip[optimize]

Functions
---------
build_network_design_model
    Build a facility location MIP.
solve_network_design
    Solve the model and extract the optimal network.

Examples
--------
>>> from deepbullwhip.chain.config import EchelonConfig
>>> from deepbullwhip.chain.graph import EdgeConfig
>>> from deepbullwhip.optimize.network_design import (
...     build_network_design_model,
...     solve_network_design,
... )
>>>
>>> candidates = {
...     "Factory_A": EchelonConfig("Factory_A", 4, 0.10, 0.40),
...     "Factory_B": EchelonConfig("Factory_B", 6, 0.08, 0.35),
...     "Warehouse": EchelonConfig("Warehouse", 2, 0.15, 0.50),
...     "Retailer": EchelonConfig("Retailer", 1, 0.20, 0.60),
... }
>>> candidate_edges = {
...     ("Factory_A", "Warehouse"): EdgeConfig(lead_time=2, transport_cost=0.05),
...     ("Factory_B", "Warehouse"): EdgeConfig(lead_time=3, transport_cost=0.03),
...     ("Warehouse", "Retailer"): EdgeConfig(lead_time=1, transport_cost=0.02),
... }
>>> fixed_costs = {"Factory_A": 1000, "Factory_B": 800, "Warehouse": 500, "Retailer": 0}
>>> demand_volume = {"Retailer": 100.0}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepbullwhip._optional import import_optional
from deepbullwhip.chain.config import EchelonConfig
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph


@dataclass
class NetworkDesignResult:
    """Results from network design optimization.

    Parameters
    ----------
    open_nodes : list[str]
        Names of facilities selected to open.
    optimal_graph : SupplyChainGraph
        The optimized supply chain graph (open nodes and active edges).
    total_cost : float
        Total cost (fixed + variable + transport).
    solver_status : str
        Solver termination condition.
    """

    open_nodes: list[str]
    optimal_graph: SupplyChainGraph
    total_cost: float
    solver_status: str


def build_network_design_model(
    candidate_nodes: dict[str, EchelonConfig],
    candidate_edges: dict[tuple[str, str], EdgeConfig],
    fixed_costs: dict[str, float],
    demand_volume: dict[str, float],
) -> Any:
    """Build a facility location MIP.

    **Decision variables**:

    - :math:`y_i \\in \\{0, 1\\}`: whether to open facility :math:`i`
    - :math:`x_{ij} \\geq 0`: flow from :math:`i` to :math:`j`

    **Objective**: Minimize total cost (fixed + holding + transport):

    .. math::

        \\min \\sum_i f_i y_i + \\sum_{(i,j)} c_{ij} x_{ij}
        + \\sum_i h_i \\cdot (\\text{throughput}_i)

    **Constraints**:

    - Demand satisfaction at each demand node
    - Flow only through open facilities
    - Flow conservation at interior nodes

    Parameters
    ----------
    candidate_nodes : dict[str, EchelonConfig]
        All candidate facility locations and their configurations.
    candidate_edges : dict[tuple[str, str], EdgeConfig]
        All candidate connections between facilities.
    fixed_costs : dict[str, float]
        Fixed cost of opening each facility.
    demand_volume : dict[str, float]
        Required demand volume at each demand-facing node.

    Returns
    -------
    pyomo.ConcreteModel
        A Pyomo MIP ready to be solved.

    Raises
    ------
    ImportError
        If ``pyomo`` is not installed.
    """
    pyo = import_optional("pyomo", "optimize")
    from pyomo.environ import (
        Binary,
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Objective,
        Param,
        Set,
        Var,
        minimize,
    )

    node_names = list(candidate_nodes.keys())
    edge_keys = list(candidate_edges.keys())
    demand_nodes = list(demand_volume.keys())

    # Identify source nodes (nodes that only supply, no incoming edges)
    has_incoming = {v for _, v in edge_keys}
    source_nodes = [n for n in node_names if n not in has_incoming]

    model = ConcreteModel("NetworkDesign")

    # Sets
    model.Nodes = Set(initialize=node_names)
    model.Edges = Set(initialize=edge_keys)
    model.DemandNodes = Set(initialize=demand_nodes)
    model.SourceNodes = Set(initialize=source_nodes)

    # Parameters
    model.fixed_cost = Param(model.Nodes, initialize=fixed_costs)
    model.holding_cost = Param(
        model.Nodes,
        initialize={n: cfg.holding_cost for n, cfg in candidate_nodes.items()},
    )
    model.transport_cost = Param(
        model.Edges,
        initialize={e: cfg.transport_cost for e, cfg in candidate_edges.items()},
    )
    model.capacity = Param(
        model.Edges,
        initialize={e: cfg.capacity for e, cfg in candidate_edges.items()},
    )
    model.demand = Param(model.DemandNodes, initialize=demand_volume)

    # Decision variables
    model.open = Var(model.Nodes, domain=Binary)
    model.flow = Var(model.Edges, domain=NonNegativeReals)

    # Objective
    def obj_rule(m):
        fixed = sum(m.fixed_cost[n] * m.open[n] for n in m.Nodes)
        transport = sum(
            m.transport_cost[e] * m.flow[e] for e in m.Edges
        )
        # Holding cost proportional to throughput
        throughput = {}
        for n in m.Nodes:
            inflow = sum(m.flow[e] for e in m.Edges if e[1] == n)
            outflow = sum(m.flow[e] for e in m.Edges if e[0] == n)
            # For demand nodes, throughput = inflow; for source, = outflow
            if n in demand_nodes:
                throughput[n] = inflow
            elif n in source_nodes:
                throughput[n] = outflow
            else:
                throughput[n] = (inflow + outflow) / 2
        holding = sum(m.holding_cost[n] * throughput[n] for n in m.Nodes)
        return fixed + transport + holding

    model.total_cost = Objective(rule=obj_rule, sense=minimize)

    # Demand satisfaction
    def demand_rule(m, n):
        inflow = sum(m.flow[e] for e in m.Edges if e[1] == n)
        return inflow >= m.demand[n]

    model.demand_met = Constraint(model.DemandNodes, rule=demand_rule)

    # Flow conservation at interior nodes
    interior = [n for n in node_names if n not in demand_nodes and n not in source_nodes]

    if interior:
        model.InteriorNodes = Set(initialize=interior)

        def conservation_rule(m, n):
            inflow = sum(m.flow[e] for e in m.Edges if e[1] == n)
            outflow = sum(m.flow[e] for e in m.Edges if e[0] == n)
            return inflow == outflow

        model.conservation = Constraint(
            model.InteriorNodes, rule=conservation_rule
        )

    # Flow only through open facilities
    big_M = sum(demand_volume.values()) * 2

    def open_constraint_src(m, i, j):
        return m.flow[i, j] <= big_M * m.open[i]

    def open_constraint_dst(m, i, j):
        return m.flow[i, j] <= big_M * m.open[j]

    model.open_src = Constraint(model.Edges, rule=open_constraint_src)
    model.open_dst = Constraint(model.Edges, rule=open_constraint_dst)

    # Capacity constraints
    def capacity_rule(m, i, j):
        return m.flow[i, j] <= m.capacity[i, j]

    model.cap = Constraint(model.Edges, rule=capacity_rule)

    # Demand nodes must be open
    def demand_open_rule(m, n):
        return m.open[n] == 1

    model.demand_open = Constraint(model.DemandNodes, rule=demand_open_rule)

    # Store references for result extraction
    model._candidate_nodes = candidate_nodes
    model._candidate_edges = candidate_edges

    return model


def solve_network_design(
    model: Any,
    solver: str = "glpk",
    time_limit: int | None = None,
    tee: bool = False,
) -> NetworkDesignResult:
    """Solve the network design model and extract the optimal network.

    Parameters
    ----------
    model : pyomo.ConcreteModel
        A model built with :func:`build_network_design_model`.
    solver : str
        Solver name (default ``"glpk"``).
    time_limit : int or None
        Maximum solve time in seconds.
    tee : bool
        If ``True``, print solver output to stdout.

    Returns
    -------
    NetworkDesignResult
        Optimal network topology and costs.

    Raises
    ------
    ImportError
        If ``pyomo`` is not installed.
    RuntimeError
        If the solver fails to find a feasible solution.
    """
    pyo = import_optional("pyomo", "optimize")
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

    # Extract open nodes
    open_nodes = [
        n for n in model.Nodes if value(model.open[n]) > 0.5
    ]

    # Build optimal graph from open nodes and active edges
    nodes = {
        n: model._candidate_nodes[n] for n in open_nodes
    }
    edges = {}
    for e in model.Edges:
        if value(model.flow[e]) > 1e-6 and e[0] in open_nodes and e[1] in open_nodes:
            edges[e] = model._candidate_edges[e]

    optimal_graph = SupplyChainGraph(nodes=nodes, edges=edges)
    total_cost = float(value(model.total_cost))

    return NetworkDesignResult(
        open_nodes=open_nodes,
        optimal_graph=optimal_graph,
        total_cost=total_cost,
        solver_status=status,
    )
