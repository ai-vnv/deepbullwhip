"""Bidirectional conversion between SupplyChainGraph and NetworkX DiGraph.

All functions in this module call :func:`~deepbullwhip._optional.import_optional`
internally, so importing this module never fails even without ``networkx``
installed -- only calling the functions does.

Examples
--------
>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.network import to_networkx, from_networkx
>>> from deepbullwhip.chain.graph import from_serial
>>>
>>> # Convert Beer Game serial config to networkx
>>> graph = from_serial(beer_game_config())
>>> G = to_networkx(graph)
>>> print(G.nodes(data="lead_time"))
>>> print(list(G.edges(data=True)))
>>>
>>> # Round-trip back
>>> graph2 = from_networkx(G)
>>> assert set(graph2.nodes) == set(graph.nodes)
"""

from __future__ import annotations

from typing import Any

from deepbullwhip._optional import import_optional
from deepbullwhip.chain.config import EchelonConfig
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph


def to_networkx(graph: SupplyChainGraph) -> Any:
    """Convert a :class:`SupplyChainGraph` to a NetworkX ``DiGraph``.

    Node attributes are populated from :class:`EchelonConfig` fields.
    Edge attributes are populated from :class:`EdgeConfig` fields.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain graph to convert.

    Returns
    -------
    nx.DiGraph
        A directed graph with node and edge attributes.

    Raises
    ------
    ImportError
        If ``networkx`` is not installed.

    Examples
    --------
    >>> from deepbullwhip.chain.graph import SupplyChainGraph, EdgeConfig, from_serial
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> G = to_networkx(from_serial(beer_game_config()))
    >>> G.number_of_nodes()
    4
    """
    nx = import_optional("networkx", "network")
    G = nx.DiGraph()

    for name, cfg in graph.nodes.items():
        G.add_node(
            name,
            lead_time=cfg.lead_time,
            holding_cost=cfg.holding_cost,
            backorder_cost=cfg.backorder_cost,
            depreciation_rate=cfg.depreciation_rate,
            service_level=cfg.service_level,
            initial_inventory=cfg.initial_inventory,
        )

    for (upstream, downstream), edge_cfg in graph.edges.items():
        G.add_edge(
            upstream,
            downstream,
            lead_time=edge_cfg.lead_time,
            capacity=edge_cfg.capacity,
            transport_cost=edge_cfg.transport_cost,
        )

    return G


def from_networkx(G: Any) -> SupplyChainGraph:
    """Convert a NetworkX ``DiGraph`` to a :class:`SupplyChainGraph`.

    Node attributes are mapped to :class:`EchelonConfig` fields.
    Missing attributes use the ``EchelonConfig`` defaults.
    Edge attributes are mapped to :class:`EdgeConfig` fields.

    Parameters
    ----------
    G : nx.DiGraph
        A directed acyclic graph. Node names become echelon names.

    Returns
    -------
    SupplyChainGraph

    Raises
    ------
    ImportError
        If ``networkx`` is not installed.
    ValueError
        If the graph is not a DAG.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_node("Factory", lead_time=4, holding_cost=0.1, backorder_cost=0.4)
    >>> G.add_node("Retailer", lead_time=1, holding_cost=0.2, backorder_cost=0.6)
    >>> G.add_edge("Factory", "Retailer", lead_time=2)
    >>> graph = from_networkx(G)
    >>> list(graph.nodes.keys())
    ['Factory', 'Retailer']
    """
    nx = import_optional("networkx", "network")

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a directed acyclic graph (DAG)")

    nodes: dict[str, EchelonConfig] = {}
    for name, attrs in G.nodes(data=True):
        nodes[str(name)] = EchelonConfig(
            name=str(name),
            lead_time=attrs.get("lead_time", 1),
            holding_cost=attrs.get("holding_cost", 0.10),
            backorder_cost=attrs.get("backorder_cost", 0.40),
            depreciation_rate=attrs.get("depreciation_rate", 0.0),
            service_level=attrs.get("service_level", 0.95),
            initial_inventory=attrs.get("initial_inventory", 50.0),
        )

    edges: dict[tuple[str, str], EdgeConfig] = {}
    for u, v, attrs in G.edges(data=True):
        edges[(str(u), str(v))] = EdgeConfig(
            lead_time=attrs.get("lead_time", 1),
            capacity=attrs.get("capacity", float("inf")),
            transport_cost=attrs.get("transport_cost", 0.0),
        )

    return SupplyChainGraph(nodes=nodes, edges=edges)


def serial_to_networkx(configs: list[EchelonConfig]) -> Any:
    """Convert a serial chain config list directly to a NetworkX ``DiGraph``.

    Convenience function combining :func:`~deepbullwhip.chain.graph.from_serial`
    and :func:`to_networkx`.

    Parameters
    ----------
    configs : list[EchelonConfig]
        Echelon configs ordered from downstream to upstream,
        matching the convention in :class:`~deepbullwhip.chain.serial.SerialSupplyChain`.

    Returns
    -------
    nx.DiGraph
        A directed graph representing the serial supply chain.

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> G = serial_to_networkx(beer_game_config())
    >>> list(G.nodes)
    ['Factory', 'Distributor', 'Wholesaler', 'Retailer']
    """
    from deepbullwhip.chain.graph import from_serial

    return to_networkx(from_serial(configs))
