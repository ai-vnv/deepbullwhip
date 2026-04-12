"""Schema data structures for supply chain network interchange.

Defines the layout and metadata types used alongside
:class:`~deepbullwhip.chain.graph.SupplyChainGraph` to produce
a complete, renderable network description.

These types are pure Python dataclasses with no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = "1.0"
"""Current version of the DeepBullwhip network JSON schema."""


@dataclass
class NodeLayoutHint:
    """Per-node layout hints for visualization.

    These hints guide the rendering engine when positioning nodes.
    All fields are optional -- the layout engine computes defaults
    from the graph topology when not specified.

    Parameters
    ----------
    tier : int or None
        Hierarchical tier (0 = source/upstream, higher = downstream).
        Auto-computed from longest path if ``None``.
    role : str
        Semantic role for styling: ``"supplier"``, ``"manufacturer"``,
        ``"distributor"``, ``"retailer"``, or custom.
    position : tuple[float, float] or None
        Explicit ``(x, y)`` position in layout coordinates.
        Overrides auto-positioning when provided.
    label : str or None
        Display label override. Defaults to the node's ``id``.
    """

    tier: int | None = None
    role: str = ""
    position: tuple[float, float] | None = None
    label: str | None = None


@dataclass
class LayoutDefaults:
    """Graph-level layout defaults.

    Controls the automatic positioning algorithm when nodes lack
    explicit positions.

    Parameters
    ----------
    orientation : str
        Layout direction: ``"TB"`` (top-to-bottom, sources at top)
        or ``"LR"`` (left-to-right, sources at left).
    tier_spacing : float
        Distance between tiers in layout units.
    node_spacing : float
        Distance between nodes within the same tier.
    auto_position : bool
        If ``True``, compute positions from topology.
        If ``False``, require explicit positions for all nodes.
    """

    orientation: str = "TB"
    tier_spacing: float = 3.0
    node_spacing: float = 3.0
    auto_position: bool = True


@dataclass
class NetworkMetadata:
    """Descriptive metadata for a supply chain network.

    Stored in JSON files but not used by the simulation engine.
    Useful for documentation, provenance, and search.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. ``"Beer Game"``).
    description : str
        Longer description of the network.
    author : str
        Author or creator.
    created : str
        ISO 8601 creation timestamp.
    tags : list[str]
        Searchable tags (e.g. ``["serial", "4-echelon"]``).
    """

    name: str = ""
    description: str = ""
    author: str = ""
    created: str = ""
    tags: list[str] = field(default_factory=list)
