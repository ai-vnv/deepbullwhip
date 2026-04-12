"""NetworkX integration for DeepBullwhip supply chain graphs.

This module provides bidirectional conversion between
:class:`~deepbullwhip.chain.graph.SupplyChainGraph` and NetworkX
``DiGraph`` objects, as well as supply-chain-specific graph analysis
functions.

Requires the ``networkx`` optional dependency::

    pip install deepbullwhip[network]

Functions
---------
to_networkx
    Convert a :class:`SupplyChainGraph` to a NetworkX ``DiGraph``.
from_networkx
    Convert a NetworkX ``DiGraph`` to a :class:`SupplyChainGraph`.
serial_to_networkx
    Convenience: convert a serial config list to a ``DiGraph``.
find_critical_path
    Longest lead-time path through the network.
echelon_centrality
    Betweenness centrality identifying bottleneck nodes.
upstream_nodes
    All ancestor nodes of a given node.
downstream_nodes
    All descendant nodes of a given node.
topological_order
    Nodes in simulation execution order (sources first).
"""

from deepbullwhip.network.analysis import (
    downstream_nodes,
    echelon_centrality,
    find_critical_path,
    topological_order,
    upstream_nodes,
)
from deepbullwhip.network.convert import (
    from_networkx,
    serial_to_networkx,
    to_networkx,
)

__all__ = [
    "to_networkx",
    "from_networkx",
    "serial_to_networkx",
    "find_critical_path",
    "echelon_centrality",
    "upstream_nodes",
    "downstream_nodes",
    "topological_order",
]
