"""Supply chain graph data model for arbitrary DAG topologies.

Provides a pure-Python representation of supply chain networks beyond
serial chains. No external dependencies required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deepbullwhip.chain.config import EchelonConfig


@dataclass
class EdgeConfig:
    """Configuration for a directed edge (link) between two echelons.

    Represents a material flow path from an upstream node to a
    downstream node.

    Parameters
    ----------
    lead_time : int
        Transport/replenishment lead time along this edge (periods).
    capacity : float
        Maximum units per period that can flow through this edge.
    transport_cost : float
        Per-unit transport cost along this edge.
    """

    lead_time: int = 1
    capacity: float = float("inf")
    transport_cost: float = 0.0


@dataclass
class SupplyChainGraph:
    """Directed acyclic graph representing a supply chain network.

    Nodes are supply chain echelons (factories, warehouses, retailers).
    Edges represent material flow from upstream (supplier) to downstream
    (customer).

    Parameters
    ----------
    nodes : dict[str, EchelonConfig]
        Mapping from node name to its configuration.
    edges : dict[tuple[str, str], EdgeConfig]
        Mapping from ``(upstream, downstream)`` to edge configuration.

    Examples
    --------
    >>> from deepbullwhip.chain.config import EchelonConfig
    >>> graph = SupplyChainGraph(
    ...     nodes={
    ...         "Factory": EchelonConfig("Factory", lead_time=4, holding_cost=0.10, backorder_cost=0.40),
    ...         "Warehouse": EchelonConfig("Warehouse", lead_time=2, holding_cost=0.15, backorder_cost=0.50),
    ...         "Retailer": EchelonConfig("Retailer", lead_time=1, holding_cost=0.20, backorder_cost=0.60),
    ...     },
    ...     edges={
    ...         ("Factory", "Warehouse"): EdgeConfig(lead_time=3),
    ...         ("Warehouse", "Retailer"): EdgeConfig(lead_time=1),
    ...     },
    ... )
    """

    nodes: dict[str, EchelonConfig] = field(default_factory=dict)
    edges: dict[tuple[str, str], EdgeConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that all edge endpoints reference existing nodes."""
        for upstream, downstream in self.edges:
            if upstream not in self.nodes:
                raise ValueError(
                    f"Edge ({upstream!r}, {downstream!r}): "
                    f"upstream node {upstream!r} not in nodes"
                )
            if downstream not in self.nodes:
                raise ValueError(
                    f"Edge ({upstream!r}, {downstream!r}): "
                    f"downstream node {downstream!r} not in nodes"
                )

    @property
    def demand_nodes(self) -> list[str]:
        """Nodes with no downstream neighbors (demand-facing / retail)."""
        has_downstream = {src for src, _ in self.edges}
        return [n for n in self.nodes if n not in has_downstream]

    @property
    def source_nodes(self) -> list[str]:
        """Nodes with no upstream neighbors (raw material sources)."""
        has_upstream = {dst for _, dst in self.edges}
        return [n for n in self.nodes if n not in has_upstream]

    def downstream_neighbors(self, node: str) -> list[str]:
        """Return nodes that receive material from *node*."""
        return [dst for src, dst in self.edges if src == node]

    def upstream_neighbors(self, node: str) -> list[str]:
        """Return nodes that supply material to *node*."""
        return [src for src, dst in self.edges if dst == node]

    def topological_order(self) -> list[str]:
        """Return nodes in topological order (sources first).

        Uses Kahn's algorithm. Raises ``ValueError`` if the graph
        contains a cycle.
        """
        in_degree: dict[str, int] = {n: 0 for n in self.nodes}
        for _, dst in self.edges:
            in_degree[dst] += 1

        queue = [n for n in self.nodes if in_degree[n] == 0]
        result: list[str] = []

        while queue:
            queue.sort()  # deterministic ordering
            node = queue.pop(0)
            result.append(node)
            for neighbor in self.downstream_neighbors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise ValueError("Supply chain graph contains a cycle")

        return result


def from_serial(configs: list[EchelonConfig]) -> SupplyChainGraph:
    """Convert a serial chain config list to a SupplyChainGraph.

    The first config is the most downstream (demand-facing) node,
    matching the convention in ``SerialSupplyChain``.

    Parameters
    ----------
    configs : list[EchelonConfig]
        Echelon configs ordered from downstream to upstream.

    Returns
    -------
    SupplyChainGraph
    """
    nodes: dict[str, EchelonConfig] = {}
    edges: dict[tuple[str, str], EdgeConfig] = {}

    for cfg in configs:
        nodes[cfg.name] = cfg

    # Connect upstream -> downstream (configs[i+1] -> configs[i])
    for i in range(len(configs) - 1):
        downstream = configs[i].name
        upstream = configs[i + 1].name
        edges[(upstream, downstream)] = EdgeConfig(
            lead_time=configs[i + 1].lead_time,
        )

    return SupplyChainGraph(nodes=nodes, edges=edges)
