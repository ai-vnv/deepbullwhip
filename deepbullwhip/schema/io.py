"""JSON serialization and deserialization for supply chain networks.

Provides lossless round-trip conversion between
:class:`~deepbullwhip.chain.graph.SupplyChainGraph` and the
DeepBullwhip JSON schema format.

No external dependencies beyond the Python standard library.

Examples
--------
>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.chain.graph import from_serial
>>> from deepbullwhip.schema.io import to_json, from_json
>>>
>>> graph = from_serial(beer_game_config())
>>> json_str = to_json(graph, metadata={"name": "Beer Game"})
>>> restored = from_json(json_str)
>>> set(restored.nodes) == set(graph.nodes)
True
"""

from __future__ import annotations

import json
import math
from typing import Any

from deepbullwhip.chain.config import EchelonConfig
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph
from deepbullwhip.schema.definition import (
    SCHEMA_VERSION,
    LayoutDefaults,
    NetworkMetadata,
    NodeLayoutHint,
)


def _serialize_value(v: Any) -> Any:
    """Convert Python values to JSON-safe values.

    ``float("inf")`` is serialized as ``null``.
    """
    if isinstance(v, float) and math.isinf(v):
        return None
    return v


def _deserialize_capacity(v: Any) -> float:
    """Convert JSON capacity value back to Python.

    ``null`` is deserialized as ``float("inf")``.
    """
    if v is None:
        return float("inf")
    return float(v)


def to_dict(
    graph: SupplyChainGraph,
    metadata: dict[str, Any] | NetworkMetadata | None = None,
    layout_hints: dict[str, NodeLayoutHint] | None = None,
    layout_defaults: LayoutDefaults | None = None,
) -> dict[str, Any]:
    """Convert a :class:`SupplyChainGraph` to a schema-compliant dictionary.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain graph to serialize.
    metadata : dict or NetworkMetadata or None
        Optional metadata to include.
    layout_hints : dict[str, NodeLayoutHint] or None
        Optional per-node layout hints.
    layout_defaults : LayoutDefaults or None
        Optional graph-level layout defaults.

    Returns
    -------
    dict
        A dictionary matching the DeepBullwhip JSON schema.
    """
    # Metadata
    if metadata is None:
        meta_dict: dict[str, Any] = {}
    elif isinstance(metadata, NetworkMetadata):
        meta_dict = {
            "name": metadata.name,
            "description": metadata.description,
            "author": metadata.author,
            "created": metadata.created,
            "tags": metadata.tags,
        }
    else:
        meta_dict = dict(metadata)

    # Nodes
    nodes_list = []
    for name, cfg in graph.nodes.items():
        node_data: dict[str, Any] = {
            "id": name,
            "config": {
                "lead_time": cfg.lead_time,
                "holding_cost": cfg.holding_cost,
                "backorder_cost": cfg.backorder_cost,
                "depreciation_rate": cfg.depreciation_rate,
                "service_level": cfg.service_level,
                "initial_inventory": cfg.initial_inventory,
            },
        }
        if layout_hints and name in layout_hints:
            hint = layout_hints[name]
            layout_data: dict[str, Any] = {}
            if hint.tier is not None:
                layout_data["tier"] = hint.tier
            if hint.role:
                layout_data["role"] = hint.role
            if hint.position is not None:
                layout_data["position"] = list(hint.position)
            if hint.label is not None:
                layout_data["label"] = hint.label
            if layout_data:
                node_data["layout"] = layout_data
        nodes_list.append(node_data)

    # Edges
    edges_list = []
    for (upstream, downstream), edge_cfg in graph.edges.items():
        edge_data: dict[str, Any] = {
            "source": upstream,
            "target": downstream,
            "config": {
                "lead_time": edge_cfg.lead_time,
                "capacity": _serialize_value(edge_cfg.capacity),
                "transport_cost": edge_cfg.transport_cost,
            },
        }
        edges_list.append(edge_data)

    result: dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "metadata": meta_dict,
        "nodes": nodes_list,
        "edges": edges_list,
    }

    if layout_defaults is not None:
        result["layout_defaults"] = {
            "orientation": layout_defaults.orientation,
            "tier_spacing": layout_defaults.tier_spacing,
            "node_spacing": layout_defaults.node_spacing,
            "auto_position": layout_defaults.auto_position,
        }

    return result


def from_dict(data: dict[str, Any]) -> SupplyChainGraph:
    """Convert a schema-compliant dictionary to a :class:`SupplyChainGraph`.

    Missing config fields use :class:`EchelonConfig` / :class:`EdgeConfig`
    defaults. ``null`` capacity values are converted to ``float("inf")``.

    Parameters
    ----------
    data : dict
        A dictionary matching the DeepBullwhip JSON schema.

    Returns
    -------
    SupplyChainGraph
    """
    nodes: dict[str, EchelonConfig] = {}
    for node_data in data.get("nodes", []):
        node_id = node_data["id"]
        cfg = node_data.get("config", {})
        nodes[node_id] = EchelonConfig(
            name=node_id,
            lead_time=cfg.get("lead_time", 1),
            holding_cost=cfg.get("holding_cost", 0.10),
            backorder_cost=cfg.get("backorder_cost", 0.40),
            depreciation_rate=cfg.get("depreciation_rate", 0.0),
            service_level=cfg.get("service_level", 0.95),
            initial_inventory=cfg.get("initial_inventory", 50.0),
        )

    edges: dict[tuple[str, str], EdgeConfig] = {}
    for edge_data in data.get("edges", []):
        source = edge_data["source"]
        target = edge_data["target"]
        cfg = edge_data.get("config", {})
        edges[(source, target)] = EdgeConfig(
            lead_time=cfg.get("lead_time", 1),
            capacity=_deserialize_capacity(cfg.get("capacity", None)),
            transport_cost=cfg.get("transport_cost", 0.0),
        )

    return SupplyChainGraph(nodes=nodes, edges=edges)


def _extract_layout_hints(data: dict[str, Any]) -> dict[str, NodeLayoutHint]:
    """Extract per-node layout hints from schema data."""
    hints: dict[str, NodeLayoutHint] = {}
    for node_data in data.get("nodes", []):
        layout = node_data.get("layout")
        if layout:
            pos = layout.get("position")
            hints[node_data["id"]] = NodeLayoutHint(
                tier=layout.get("tier"),
                role=layout.get("role", ""),
                position=tuple(pos) if pos else None,
                label=layout.get("label"),
            )
    return hints


def _extract_metadata(data: dict[str, Any]) -> NetworkMetadata:
    """Extract metadata from schema data."""
    meta = data.get("metadata", {})
    return NetworkMetadata(
        name=meta.get("name", ""),
        description=meta.get("description", ""),
        author=meta.get("author", ""),
        created=meta.get("created", ""),
        tags=meta.get("tags", []),
    )


def _extract_layout_defaults(data: dict[str, Any]) -> LayoutDefaults:
    """Extract layout defaults from schema data."""
    defaults = data.get("layout_defaults", {})
    return LayoutDefaults(
        orientation=defaults.get("orientation", "TB"),
        tier_spacing=defaults.get("tier_spacing", 2.0),
        node_spacing=defaults.get("node_spacing", 3.0),
        auto_position=defaults.get("auto_position", True),
    )


def to_json(
    graph: SupplyChainGraph,
    metadata: dict[str, Any] | NetworkMetadata | None = None,
    layout_hints: dict[str, NodeLayoutHint] | None = None,
    layout_defaults: LayoutDefaults | None = None,
    indent: int = 2,
) -> str:
    """Serialize a :class:`SupplyChainGraph` to a JSON string.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain graph to serialize.
    metadata : dict or NetworkMetadata or None
        Optional metadata.
    layout_hints : dict[str, NodeLayoutHint] or None
        Optional per-node layout hints.
    layout_defaults : LayoutDefaults or None
        Optional layout defaults.
    indent : int
        JSON indentation level (default 2).

    Returns
    -------
    str
        A JSON string matching the DeepBullwhip schema.

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> from deepbullwhip.chain.graph import from_serial
    >>> json_str = to_json(from_serial(beer_game_config()))
    >>> '"version": "1.0"' in json_str
    True
    """
    data = to_dict(graph, metadata, layout_hints, layout_defaults)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def from_json(json_str: str) -> SupplyChainGraph:
    """Deserialize a JSON string to a :class:`SupplyChainGraph`.

    Parameters
    ----------
    json_str : str
        A JSON string matching the DeepBullwhip schema.

    Returns
    -------
    SupplyChainGraph

    Examples
    --------
    >>> graph = from_json('{"nodes": [{"id": "A", "config": {"lead_time": 1, "holding_cost": 0.1, "backorder_cost": 0.5}}], "edges": []}')
    >>> "A" in graph.nodes
    True
    """
    data = json.loads(json_str)
    return from_dict(data)


def save_json(
    graph: SupplyChainGraph,
    path: str,
    metadata: dict[str, Any] | NetworkMetadata | None = None,
    layout_hints: dict[str, NodeLayoutHint] | None = None,
    layout_defaults: LayoutDefaults | None = None,
    indent: int = 2,
) -> None:
    """Write a supply chain graph to a JSON file.

    Parameters
    ----------
    graph : SupplyChainGraph
        The graph to save.
    path : str
        Output file path.
    metadata : dict or NetworkMetadata or None
        Optional metadata.
    layout_hints : dict[str, NodeLayoutHint] or None
        Optional layout hints.
    layout_defaults : LayoutDefaults or None
        Optional layout defaults.
    indent : int
        JSON indentation (default 2).
    """
    json_str = to_json(graph, metadata, layout_hints, layout_defaults, indent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)


def load_json(path: str) -> SupplyChainGraph:
    """Load a supply chain graph from a JSON file.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    SupplyChainGraph
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return from_dict(data)


def load_json_full(
    path: str,
) -> tuple[SupplyChainGraph, NetworkMetadata, dict[str, NodeLayoutHint]]:
    """Load a graph with metadata and layout hints from a JSON file.

    Returns the full parsed content for use with the rendering API.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    tuple[SupplyChainGraph, NetworkMetadata, dict[str, NodeLayoutHint]]
        The graph, metadata, and per-node layout hints.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    graph = from_dict(data)
    metadata = _extract_metadata(data)
    layout_hints = _extract_layout_hints(data)
    return graph, metadata, layout_hints
