"""Publication theme system for supply chain network rendering.

Defines frozen dataclass-based themes that control every visual aspect
of rendered supply chain diagrams. Four built-in themes are provided,
and custom themes can be registered via :func:`register_theme`.

Themes are composable: use :meth:`Theme.override` to create variants
with selected fields replaced.

Built-in Themes
---------------
``kfupm``
    KFUPM AI V&V Lab palette (green/gold/teal). Serif fonts, 7" width.
    Default theme for journal papers.
``ieee``
    IEEE single-column format. Grayscale-friendly, serif 7pt, 3.5" width.
``presentation``
    High contrast for slides. Sans-serif 12pt, 10" width.
``minimal``
    Clean black-and-white. No fills, thin lines, serif fonts.

Examples
--------
>>> from deepbullwhip.render.theme import get_theme, list_themes
>>> theme = get_theme("ieee")
>>> theme.figure.width
3.5
>>> list_themes()
['kfupm', 'ieee', 'presentation', 'minimal']

>>> # Compose a custom variant
>>> my_theme = get_theme("kfupm").override(
...     font=FontStyle(family="sans-serif", node_label_size=10.0)
... )
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeStyle:
    """Visual style for supply chain nodes.

    Parameters
    ----------
    shape : str
        Node shape: ``"rounded_rect"``, ``"rect"``, ``"ellipse"``.
    fill_alpha : float
        Fill opacity in ``[0, 1]``.
    border_width : float
        Border line width in points.
    min_width : float
        Minimum node width (inches for matplotlib, cm for TikZ).
    min_height : float
        Minimum node height.
    corner_radius : float
        Corner radius for rounded shapes.
    """

    shape: str = "rounded_rect"
    fill_alpha: float = 0.90
    border_width: float = 1.0
    min_width: float = 0.85
    min_height: float = 0.45
    corner_radius: float = 0.06


@dataclass(frozen=True)
class EdgeStyle:
    """Visual style for supply chain edges (material flow links).

    Parameters
    ----------
    arrow_style : str
        Arrow head style: ``"-|>"``, ``"->"``, ``"->>"``.
    line_width : float
        Edge line width in points.
    color : str
        Edge color as hex string.
    curve_radius : float
        Arc curvature (0 = straight lines).
    """

    arrow_style: str = "-|>"
    line_width: float = 1.2
    color: str = "#888888"
    curve_radius: float = 0.0


@dataclass(frozen=True)
class FontStyle:
    """Typography settings for supply chain diagrams.

    Parameters
    ----------
    family : str
        Font family: ``"serif"`` or ``"sans-serif"``.
    node_label_size : float
        Font size for node names (points).
    node_detail_size : float
        Font size for node details (config values, metrics).
    edge_label_size : float
        Font size for edge labels.
    title_size : float
        Font size for the figure title.
    title_weight : str
        Font weight for the title: ``"bold"`` or ``"normal"``.
    """

    family: str = "serif"
    node_label_size: float = 8.0
    node_detail_size: float = 6.5
    edge_label_size: float = 7.0
    title_size: float = 12.0
    title_weight: str = "bold"


@dataclass(frozen=True)
class FigureStyle:
    """Figure-level layout settings.

    Parameters
    ----------
    width : float
        Figure width in inches.
    height : float or None
        Figure height. If ``None``, auto-computed from content.
    dpi : int
        Resolution for raster output (PNG).
    background : str
        Background color as hex string.
    margin : float
        Margin around content in inches.
    """

    width: float = 7.0
    height: float | None = None
    dpi: int = 300
    background: str = "white"
    margin: float = 0.5


@dataclass(frozen=True)
class Theme:
    """Complete visual theme for supply chain diagram rendering.

    A theme controls every visual aspect: colors, fonts, shapes, sizing.
    Themes are immutable -- use :meth:`override` to create variants.

    Parameters
    ----------
    name : str
        Theme identifier.
    colors : list[str]
        Echelon color cycle (hex strings). Applied cyclically to nodes.
    bw_colors : dict[str, str]
        Bullwhip ratio severity colors:
        ``{"low": ..., "medium": ..., "high": ...}``.
    node : NodeStyle
        Node visual style.
    edge : EdgeStyle
        Edge visual style.
    font : FontStyle
        Typography settings.
    figure : FigureStyle
        Figure-level settings.

    Examples
    --------
    >>> theme = Theme(
    ...     name="custom",
    ...     colors=["#FF0000", "#00FF00"],
    ...     bw_colors={"low": "#00FF00", "medium": "#FFFF00", "high": "#FF0000"},
    ... )
    >>> theme.colors[0]
    '#FF0000'
    """

    name: str = "custom"
    colors: list[str] = field(
        default_factory=lambda: ["#006747", "#C4972F", "#2E8B8B", "#8B4513"]
    )
    bw_colors: dict[str, str] = field(
        default_factory=lambda: {
            "low": "#2E8B57",
            "medium": "#DAA520",
            "high": "#CD5C5C",
        }
    )
    node: NodeStyle = field(default_factory=NodeStyle)
    edge: EdgeStyle = field(default_factory=EdgeStyle)
    font: FontStyle = field(default_factory=FontStyle)
    figure: FigureStyle = field(default_factory=FigureStyle)

    def override(self, **kwargs: Any) -> Theme:
        """Return a new theme with selected fields replaced.

        Parameters
        ----------
        **kwargs
            Fields to replace (e.g. ``font=FontStyle(...)``).

        Returns
        -------
        Theme
            A new theme instance with the specified overrides.

        Examples
        --------
        >>> theme = get_theme("kfupm")
        >>> big = theme.override(font=FontStyle(node_label_size=14.0))
        >>> big.font.node_label_size
        14.0
        >>> big.name  # unchanged fields preserved
        'kfupm'
        """
        return dataclasses.replace(self, **kwargs)

    def node_color(self, index: int) -> str:
        """Return the color for the node at *index* (cycles through palette).

        Parameters
        ----------
        index : int
            Node index.

        Returns
        -------
        str
            Hex color string.
        """
        return self.colors[index % len(self.colors)]

    def bw_color(self, ratio: float) -> str:
        """Return a color based on bullwhip ratio severity.

        Parameters
        ----------
        ratio : float
            Bullwhip ratio value.

        Returns
        -------
        str
            Hex color string (green for low, gold for medium, red for high).
        """
        if ratio < 1.5:
            return self.bw_colors["low"]
        elif ratio < 3.0:
            return self.bw_colors["medium"]
        else:
            return self.bw_colors["high"]


# ── Theme Registry ──────────────────────────────────────────────────

_THEMES: dict[str, Theme] = {}


def register_theme(name: str, theme: Theme) -> None:
    """Register a named theme for use with :func:`get_theme`.

    Parameters
    ----------
    name : str
        Theme name.
    theme : Theme
        Theme instance to register.
    """
    _THEMES[name] = theme


def get_theme(name: str) -> Theme:
    """Retrieve a registered theme by name.

    Parameters
    ----------
    name : str
        Theme name (e.g. ``"kfupm"``, ``"ieee"``).

    Returns
    -------
    Theme

    Raises
    ------
    KeyError
        If the theme name is not registered.
    """
    if name not in _THEMES:
        available = list(_THEMES.keys())
        raise KeyError(
            f"Theme '{name}' not found. Available: {available}"
        )
    return _THEMES[name]


def list_themes() -> list[str]:
    """List all registered theme names.

    Returns
    -------
    list[str]
    """
    return list(_THEMES.keys())


# ── Built-in Themes ─────────────────────────────────────────────────

KFUPM_THEME = Theme(
    name="kfupm",
    colors=["#006747", "#C4972F", "#2E8B8B", "#8B4513", "#4169E1", "#8B008B"],
    bw_colors={"low": "#2E8B57", "medium": "#DAA520", "high": "#CD5C5C"},
    node=NodeStyle(shape="circle", fill_alpha=0.90, border_width=1.0, min_width=1.1),
    edge=EdgeStyle(color="#333333", line_width=1.0),
    font=FontStyle(family="sans-serif", node_label_size=8.0, node_detail_size=6.0,
                   edge_label_size=7.5),
    figure=FigureStyle(width=7.0, dpi=300),
)

IEEE_THEME = Theme(
    name="ieee",
    colors=["#333333", "#666666", "#999999", "#BBBBBB", "#444444", "#777777"],
    bw_colors={"low": "#555555", "medium": "#888888", "high": "#333333"},
    node=NodeStyle(shape="circle", fill_alpha=0.85, border_width=0.8, min_width=0.9),
    edge=EdgeStyle(color="#444444", line_width=0.8),
    font=FontStyle(
        family="serif", node_label_size=6.5, node_detail_size=5.0,
        edge_label_size=6.0, title_size=9.0,
    ),
    figure=FigureStyle(width=3.5, dpi=300),
)

PRESENTATION_THEME = Theme(
    name="presentation",
    colors=["#006747", "#C4972F", "#2E8B8B", "#8B4513", "#4169E1", "#8B008B"],
    bw_colors={"low": "#2E8B57", "medium": "#DAA520", "high": "#CD5C5C"},
    node=NodeStyle(shape="circle", fill_alpha=0.90, border_width=1.5, min_width=1.5),
    edge=EdgeStyle(color="#555555", line_width=1.5),
    font=FontStyle(
        family="sans-serif", node_label_size=11.0, node_detail_size=8.0,
        edge_label_size=9.0, title_size=16.0,
    ),
    figure=FigureStyle(width=10.0, dpi=150),
)

MINIMAL_THEME = Theme(
    name="minimal",
    colors=["#555555", "#777777", "#999999", "#BBBBBB", "#444444", "#666666"],
    bw_colors={"low": "#888888", "medium": "#555555", "high": "#222222"},
    node=NodeStyle(shape="circle", fill_alpha=0.80, border_width=0.8, min_width=1.0),
    edge=EdgeStyle(color="#444444", line_width=0.8, arrow_style="->"),
    font=FontStyle(family="sans-serif", node_label_size=7.5, node_detail_size=5.5),
    figure=FigureStyle(width=7.0, dpi=300),
)

# Register built-in themes
for _theme in (KFUPM_THEME, IEEE_THEME, PRESENTATION_THEME, MINIMAL_THEME):
    register_theme(_theme.name, _theme)
