"""Unified rendering API for supply chain network diagrams.

Provides a single entry point :func:`render_graph` that dispatches
to the appropriate backend (matplotlib, Graphviz, or TikZ) while
applying a consistent theme and layout.

Functions
---------
render_graph
    Render a supply chain graph with any backend and theme.
render_from_json
    Load a JSON file and render in one call.

Examples
--------
Render with matplotlib (default):

>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.chain.graph import from_serial
>>> from deepbullwhip.render.api import render_graph
>>>
>>> graph = from_serial(beer_game_config())
>>> fig = render_graph(graph, backend="matplotlib", theme="kfupm")
>>> fig.savefig("beer_game.pdf")  # doctest: +SKIP

Render as TikZ for LaTeX papers:

>>> tex = render_graph(graph, backend="tikz", theme="ieee")
>>> with open("beer_game.tex", "w") as f:
...     f.write(tex)  # doctest: +SKIP

Render from JSON with Graphviz:

>>> source = render_from_json(
...     "beer_game.json", backend="graphviz", theme="minimal"
... )  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Literal

from deepbullwhip._types import NetworkSimulationResult, SimulationResult
from deepbullwhip.chain.graph import SupplyChainGraph
from deepbullwhip.render.layout import compute_positions
from deepbullwhip.render.theme import Theme, get_theme
from deepbullwhip.schema.definition import LayoutDefaults, NodeLayoutHint


def render_graph(
    graph: SupplyChainGraph,
    backend: Literal["matplotlib", "graphviz", "tikz"] = "matplotlib",
    theme: str | Theme = "kfupm",
    sim_result: SimulationResult | NetworkSimulationResult | None = None,
    layout_hints: dict[str, NodeLayoutHint] | None = None,
    layout_defaults: LayoutDefaults | None = None,
    title: str | None = None,
    annotations: dict[str, dict[str, str]] | None = None,
    **backend_kwargs: Any,
) -> Any:
    """Render a supply chain graph with the specified backend and theme.

    This is the primary entry point for the standardized rendering
    system. It computes positions from the graph topology, applies
    the theme, and dispatches to the selected backend.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology to render.
    backend : ``"matplotlib"`` | ``"graphviz"`` | ``"tikz"``
        Rendering backend:

        - ``"matplotlib"``: returns ``matplotlib.figure.Figure``
        - ``"graphviz"``: returns ``graphviz.Source`` (requires ``pip install deepbullwhip[viz]``)
        - ``"tikz"``: returns ``str`` (LaTeX/TikZ source code)
    theme : str or Theme
        Theme name (e.g. ``"kfupm"``, ``"ieee"``) or a :class:`Theme` instance.
    sim_result : SimulationResult or NetworkSimulationResult or None
        Optional simulation results for metric overlay on nodes.
    layout_hints : dict[str, NodeLayoutHint] or None
        Per-node layout overrides (tier, position, label).
    layout_defaults : LayoutDefaults or None
        Graph-level layout settings (orientation, spacing).
    title : str or None
        Figure title.
    annotations : dict[str, dict[str, str]] or None
        Extra per-node text annotations (matplotlib only).
    **backend_kwargs
        Backend-specific options:

        - **graphviz**: ``engine`` (``"dot"``, ``"neato"``), ``fmt`` (``"svg"``, ``"pdf"``)
        - **tikz**: ``standalone`` (``True``/``False``)

    Returns
    -------
    matplotlib.figure.Figure or graphviz.Source or str
        The rendered output, depending on the backend.

    Raises
    ------
    ValueError
        If an unknown backend is specified.
    ImportError
        If the graphviz backend is requested but ``graphviz`` is not installed.

    Examples
    --------
    **2-tier consumer chain (matplotlib):**

    >>> from deepbullwhip.chain.config import consumer_2tier_config
    >>> from deepbullwhip.chain.graph import from_serial
    >>> fig = render_graph(from_serial(consumer_2tier_config()), theme="minimal")

    **4-tier beer game (TikZ for IEEE paper):**

    >>> from deepbullwhip.chain.config import beer_game_config
    >>> tex = render_graph(
    ...     from_serial(beer_game_config()),
    ...     backend="tikz",
    ...     theme="ieee",
    ...     title="Beer Game Supply Chain",
    ... )

    **Distribution tree (Graphviz):**

    >>> from deepbullwhip.chain.graph import SupplyChainGraph, EdgeConfig
    >>> from deepbullwhip.chain.config import EchelonConfig
    >>> tree = SupplyChainGraph(
    ...     nodes={
    ...         "Factory": EchelonConfig("Factory", 4, 0.10, 0.40),
    ...         "WH": EchelonConfig("WH", 2, 0.15, 0.50),
    ...         "Store_A": EchelonConfig("Store_A", 1, 0.20, 0.60),
    ...         "Store_B": EchelonConfig("Store_B", 1, 0.20, 0.60),
    ...     },
    ...     edges={
    ...         ("Factory", "WH"): EdgeConfig(lead_time=3),
    ...         ("WH", "Store_A"): EdgeConfig(lead_time=1),
    ...         ("WH", "Store_B"): EdgeConfig(lead_time=1),
    ...     },
    ... )
    >>> source = render_graph(tree, backend="graphviz", engine="dot")  # doctest: +SKIP
    """
    # Resolve theme
    if isinstance(theme, str):
        resolved_theme = get_theme(theme)
    else:
        resolved_theme = theme

    # Compute positions
    positions = compute_positions(graph, layout_hints, layout_defaults)

    # Dispatch to backend
    if backend == "matplotlib":
        from deepbullwhip.render._matplotlib import render_matplotlib

        return render_matplotlib(
            graph, positions, resolved_theme, sim_result, title, annotations
        )
    elif backend == "graphviz":
        from deepbullwhip.render._graphviz import render_graphviz

        return render_graphviz(
            graph, positions, resolved_theme, sim_result, title, **backend_kwargs
        )
    elif backend == "tikz":
        from deepbullwhip.render._tikz import render_tikz

        return render_tikz(
            graph, positions, resolved_theme, sim_result, title, **backend_kwargs
        )
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Choose from: 'matplotlib', 'graphviz', 'tikz'"
        )


def render_from_json(
    json_path: str,
    backend: Literal["matplotlib", "graphviz", "tikz"] = "matplotlib",
    theme: str | Theme = "kfupm",
    **kwargs: Any,
) -> Any:
    """Load a JSON network file and render in one call.

    Convenience function that combines :func:`~deepbullwhip.schema.io.load_json_full`
    with :func:`render_graph`.

    Parameters
    ----------
    json_path : str
        Path to a DeepBullwhip JSON schema file.
    backend : ``"matplotlib"`` | ``"graphviz"`` | ``"tikz"``
        Rendering backend.
    theme : str or Theme
        Theme name or instance.
    **kwargs
        Additional arguments passed to :func:`render_graph`.

    Returns
    -------
    matplotlib.figure.Figure or graphviz.Source or str
    """
    from deepbullwhip.schema.io import load_json_full

    graph, _metadata, layout_hints = load_json_full(json_path)
    return render_graph(
        graph,
        backend=backend,
        theme=theme,
        layout_hints=layout_hints if layout_hints else None,
        **kwargs,
    )
