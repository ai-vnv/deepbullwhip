"""Supply-chain-specific graph analysis functions using NetworkX.

All functions accept a NetworkX ``DiGraph`` (as returned by
:func:`~deepbullwhip.network.convert.to_networkx`) and provide
supply-chain-meaningful interpretations of standard graph algorithms.

Examples
--------
>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.network import serial_to_networkx, find_critical_path
>>> G = serial_to_networkx(beer_game_config())
>>> find_critical_path(G)
['Factory', 'Distributor', 'Wholesaler', 'Retailer']
"""

from __future__ import annotations

from typing import Any

from deepbullwhip._optional import import_optional


def find_critical_path(G: Any) -> list[str]:
    """Find the longest lead-time path through the supply chain.

    The critical path determines the end-to-end replenishment lead
    time. Edge weights are ``lead_time`` attributes (default 1).

    Parameters
    ----------
    G : nx.DiGraph
        A directed acyclic supply chain graph.

    Returns
    -------
    list[str]
        Node names along the longest path, from source to sink.

    Raises
    ------
    ImportError
        If ``networkx`` is not installed.

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> from deepbullwhip.network import serial_to_networkx
    >>> G = serial_to_networkx(beer_game_config())
    >>> path = find_critical_path(G)
    >>> len(path)
    4
    """
    nx = import_optional("networkx", "network")

    # Use negative weights to find longest path via shortest-path algorithm
    longest_path = nx.dag_longest_path(G, weight="lead_time")
    return [str(n) for n in longest_path]


def critical_path_length(G: Any) -> float:
    """Total lead time along the critical (longest) path.

    Parameters
    ----------
    G : nx.DiGraph
        A directed acyclic supply chain graph.

    Returns
    -------
    float
        Sum of ``lead_time`` edge weights along the longest path.

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> from deepbullwhip.network import serial_to_networkx
    >>> G = serial_to_networkx(beer_game_config())
    >>> critical_path_length(G)
    6
    """
    nx = import_optional("networkx", "network")
    return nx.dag_longest_path_length(G, weight="lead_time")


def echelon_centrality(G: Any) -> dict[str, float]:
    """Betweenness centrality of supply chain nodes.

    Higher centrality indicates a bottleneck node -- disruption there
    affects more supply paths.

    Parameters
    ----------
    G : nx.DiGraph
        A directed acyclic supply chain graph.

    Returns
    -------
    dict[str, float]
        Mapping from node name to centrality score in ``[0, 1]``.

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> from deepbullwhip.network import serial_to_networkx
    >>> G = serial_to_networkx(beer_game_config())
    >>> centrality = echelon_centrality(G)
    >>> isinstance(centrality, dict)
    True
    """
    nx = import_optional("networkx", "network")
    raw = nx.betweenness_centrality(G)
    return {str(k): float(v) for k, v in raw.items()}


def upstream_nodes(G: Any, node: str) -> set[str]:
    """All ancestor nodes of *node* in the supply chain.

    These are all suppliers (direct and indirect) that feed into
    the given node.

    Parameters
    ----------
    G : nx.DiGraph
        A directed acyclic supply chain graph.
    node : str
        The node to query.

    Returns
    -------
    set[str]
        Set of ancestor node names (not including *node* itself).

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> from deepbullwhip.network import serial_to_networkx
    >>> G = serial_to_networkx(beer_game_config())
    >>> upstream_nodes(G, "Retailer")
    {'Wholesaler', 'Distributor', 'Factory'}
    """
    nx = import_optional("networkx", "network")
    return {str(n) for n in nx.ancestors(G, node)}


def downstream_nodes(G: Any, node: str) -> set[str]:
    """All descendant nodes of *node* in the supply chain.

    These are all customers (direct and indirect) served by
    the given node.

    Parameters
    ----------
    G : nx.DiGraph
        A directed acyclic supply chain graph.
    node : str
        The node to query.

    Returns
    -------
    set[str]
        Set of descendant node names (not including *node* itself).

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> from deepbullwhip.network import serial_to_networkx
    >>> G = serial_to_networkx(beer_game_config())
    >>> downstream_nodes(G, "Factory")
    {'Distributor', 'Wholesaler', 'Retailer'}
    """
    nx = import_optional("networkx", "network")
    return {str(n) for n in nx.descendants(G, node)}


def topological_order(G: Any) -> list[str]:
    """Return nodes in topological order (sources first).

    This is the simulation execution order: upstream nodes are
    processed before downstream nodes.

    Parameters
    ----------
    G : nx.DiGraph
        A directed acyclic supply chain graph.

    Returns
    -------
    list[str]
        Nodes sorted in topological order.

    Raises
    ------
    NetworkXUnfeasible
        If the graph contains a cycle.

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> from deepbullwhip.network import serial_to_networkx
    >>> G = serial_to_networkx(beer_game_config())
    >>> order = topological_order(G)
    >>> order[0]
    'Factory'
    """
    nx = import_optional("networkx", "network")
    return [str(n) for n in nx.topological_sort(G)]
