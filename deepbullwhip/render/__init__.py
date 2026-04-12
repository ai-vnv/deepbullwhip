"""Multi-backend supply chain network renderer with theme support.

Provides a unified API for rendering :class:`SupplyChainGraph` objects
as publication-quality diagrams across three backends:

- **matplotlib** (PNG/PDF): for Jupyter notebooks and inline display
- **Graphviz** (SVG/PDF): for complex layouts via ``dot``/``neato``
- **TikZ** (LaTeX): for direct paper integration

All backends produce visually consistent output driven by a shared
theme system with four built-in themes.

Quick Start
-----------
>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.chain.graph import from_serial
>>> from deepbullwhip.render import render_graph
>>>
>>> graph = from_serial(beer_game_config())
>>> fig = render_graph(graph, theme="kfupm")
>>> fig.savefig("beer_game.pdf")  # doctest: +SKIP
"""

from deepbullwhip.render.api import render_from_json, render_graph
from deepbullwhip.render.theme import (
    EdgeStyle,
    FigureStyle,
    FontStyle,
    NodeStyle,
    Theme,
    get_theme,
    list_themes,
    register_theme,
)

__all__ = [
    "render_graph",
    "render_from_json",
    "Theme",
    "NodeStyle",
    "EdgeStyle",
    "FontStyle",
    "FigureStyle",
    "get_theme",
    "list_themes",
    "register_theme",
]
