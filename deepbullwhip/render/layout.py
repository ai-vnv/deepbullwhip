"""Automatic layout computation for supply chain network diagrams.

Computes node positions from graph topology using a tier-based
hierarchical algorithm. Source nodes are placed at the top (TB) or
left (LR), with downstream nodes at progressively lower tiers.

The layout algorithm ensures clean, Bayesian-network-style diagrams
with evenly spaced nodes and aligned tiers.

Functions
---------
compute_tiers
    Assign tier numbers from topology (longest path from sources).
compute_positions
    Compute ``(x, y)`` positions for all nodes.
compute_figure_size
    Compute figure dimensions from positions and theme.

Examples
--------
>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.chain.graph import from_serial
>>> from deepbullwhip.render.layout import compute_positions
>>>
>>> graph = from_serial(beer_game_config())
>>> positions = compute_positions(graph)
>>> len(positions) == 4
True
"""

from __future__ import annotations

from deepbullwhip.chain.graph import SupplyChainGraph
from deepbullwhip.render.theme import Theme
from deepbullwhip.schema.definition import LayoutDefaults, NodeLayoutHint


def compute_tiers(graph: SupplyChainGraph) -> dict[str, int]:
    """Assign tier numbers based on longest path from source nodes.

    Source nodes (no incoming edges) are tier 0. Each node's tier
    is 1 + max(tier of upstream neighbors). This produces a
    hierarchical layout where upstream nodes appear before downstream.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain graph.

    Returns
    -------
    dict[str, int]
        Mapping from node name to tier number.

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> from deepbullwhip.chain.graph import from_serial
    >>> tiers = compute_tiers(from_serial(beer_game_config()))
    >>> tiers["Factory"]
    0
    """
    topo_order = graph.topological_order()
    tiers: dict[str, int] = {}

    for node in topo_order:
        upstream = graph.upstream_neighbors(node)
        if not upstream:
            tiers[node] = 0
        else:
            tiers[node] = max(tiers[u] for u in upstream) + 1

    return tiers


def compute_positions(
    graph: SupplyChainGraph,
    layout_hints: dict[str, NodeLayoutHint] | None = None,
    defaults: LayoutDefaults | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute ``(x, y)`` positions for all nodes.

    Uses a tier-based hierarchical layout:

    1. Compute tiers via :func:`compute_tiers`.
    2. Group nodes by tier.
    3. Space nodes within each tier evenly.
    4. Override with explicit positions from ``layout_hints``.

    For ``"TB"`` orientation, tier 0 is at the top (highest y).
    For ``"LR"`` orientation, tier 0 is at the left.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain graph.
    layout_hints : dict[str, NodeLayoutHint] or None
        Per-node layout overrides. Nodes with explicit ``position``
        fields skip auto-positioning.
    defaults : LayoutDefaults or None
        Graph-level layout settings. Uses sensible defaults if ``None``.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from node name to ``(x, y)`` position.

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> from deepbullwhip.chain.graph import from_serial
    >>> pos = compute_positions(from_serial(beer_game_config()))
    >>> pos["Factory"][1] > pos["Retailer"][1]  # Factory higher in TB
    True
    """
    if defaults is None:
        defaults = LayoutDefaults()

    # Use hints for tier overrides
    tiers = compute_tiers(graph)
    if layout_hints:
        for name, hint in layout_hints.items():
            if hint.tier is not None and name in tiers:
                tiers[name] = hint.tier

    # Group nodes by tier
    tier_groups: dict[int, list[str]] = {}
    for node, tier in tiers.items():
        tier_groups.setdefault(tier, []).append(node)

    # Sort within tiers for deterministic layout
    for tier in tier_groups:
        tier_groups[tier].sort()

    max_tier = max(tier_groups.keys()) if tier_groups else 0

    positions: dict[str, tuple[float, float]] = {}

    for tier, nodes_in_tier in tier_groups.items():
        n = len(nodes_in_tier)
        # Center nodes within the tier
        total_width = (n - 1) * defaults.node_spacing
        start_offset = -total_width / 2

        for i, node_name in enumerate(nodes_in_tier):
            cross_pos = start_offset + i * defaults.node_spacing
            tier_pos = (max_tier - tier) * defaults.tier_spacing

            if defaults.orientation == "TB":
                positions[node_name] = (cross_pos, tier_pos)
            else:  # LR
                positions[node_name] = (tier * defaults.tier_spacing, cross_pos)

    # Override with explicit positions
    if layout_hints:
        for name, hint in layout_hints.items():
            if hint.position is not None and name in positions:
                positions[name] = hint.position

    return positions


def compute_figure_size(
    positions: dict[str, tuple[float, float]],
    theme: Theme,
) -> tuple[float, float]:
    """Compute figure dimensions from node positions and theme.

    Parameters
    ----------
    positions : dict[str, tuple[float, float]]
        Node positions.
    theme : Theme
        The rendering theme (provides width, margin, height hint).

    Returns
    -------
    tuple[float, float]
        ``(width, height)`` in inches.
    """
    if theme.figure.height is not None:
        return (theme.figure.width, theme.figure.height)

    if not positions:
        return (theme.figure.width, theme.figure.width * 0.6)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    x_range = max(xs) - min(xs) if len(xs) > 1 else 1.0
    y_range = max(ys) - min(ys) if len(ys) > 1 else 1.0

    aspect = (y_range + 2 * theme.figure.margin) / (x_range + 2 * theme.figure.margin)
    aspect = max(0.4, min(aspect, 1.5))  # clamp

    width = theme.figure.width
    height = width * aspect
    return (width, height)
