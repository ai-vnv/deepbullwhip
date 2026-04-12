"""Publication-quality supply chain visualization using Graphviz.

Renders :class:`~deepbullwhip.chain.graph.SupplyChainGraph` objects as
Graphviz DOT graphs, with optional simulation result overlays showing
bullwhip ratios, inventory levels, and material flows.

Complements the matplotlib-based visualizations in
:mod:`deepbullwhip.diagnostics.network`. Use Graphviz for:

- Hierarchical layouts with automatic edge routing (``dot`` engine)
- SVG/PDF export for LaTeX papers
- Complex DAG topologies where matplotlib node placement is awkward

Requires the ``graphviz`` optional dependency::

    pip install deepbullwhip[viz]

Functions
---------
render_network
    Render the full supply chain network with optional metrics overlay.
render_simulation_snapshot
    Render network state at a specific simulation period.
save_figure
    Save a rendered graph to SVG, PDF, or PNG.

Examples
--------
>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.chain.graph import from_serial
>>> from deepbullwhip.diagnostics.graphviz_viz import render_network
>>>
>>> graph = from_serial(beer_game_config())
>>> source = render_network(graph)
>>> source.render("beer_game", format="pdf", cleanup=True)
"""

from __future__ import annotations

from typing import Any

from deepbullwhip._optional import import_optional
from deepbullwhip._types import NetworkSimulationResult, SimulationResult
from deepbullwhip.chain.graph import SupplyChainGraph

# KFUPM-inspired color palette (matching diagnostics/plots.py)
_COLORS = ["#006747", "#C4972F", "#2E8B8B", "#8B4513", "#4169E1", "#8B008B"]
_BW_COLORS = {
    "low": "#2E8B57",     # green: BWR < 1.5
    "medium": "#DAA520",  # gold: 1.5 <= BWR < 3.0
    "high": "#CD5C5C",    # red: BWR >= 3.0
}


def _bw_color(ratio: float) -> str:
    """Return a color code based on bullwhip ratio severity."""
    if ratio < 1.5:
        return _BW_COLORS["low"]
    elif ratio < 3.0:
        return _BW_COLORS["medium"]
    else:
        return _BW_COLORS["high"]


def _echelon_color(index: int) -> str:
    """Return a color for the echelon at *index*."""
    return _COLORS[index % len(_COLORS)]


def render_network(
    graph: SupplyChainGraph,
    sim_result: SimulationResult | NetworkSimulationResult | None = None,
    engine: str = "dot",
    fmt: str = "svg",
    rankdir: str = "LR",
    title: str | None = None,
) -> Any:
    """Render the supply chain network as a Graphviz ``Source`` object.

    Nodes are drawn as rounded boxes showing echelon configuration.
    When ``sim_result`` is provided, nodes are color-coded by bullwhip
    ratio and annotated with simulation metrics (BWR, fill rate, cost).

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology to render.
    sim_result : SimulationResult or NetworkSimulationResult or None
        If provided, overlay simulation metrics on the graph.
    engine : str
        Graphviz layout engine. Common choices:

        - ``"dot"`` (default): hierarchical left-to-right or top-to-bottom
        - ``"neato"``: force-directed, good for complex DAGs
        - ``"fdp"``: force-directed for larger networks
    fmt : str
        Output format: ``"svg"``, ``"pdf"``, ``"png"``.
    rankdir : str
        Layout direction: ``"LR"`` (left-to-right) or ``"TB"``
        (top-to-bottom).
    title : str or None
        Optional title displayed at the top of the graph.

    Returns
    -------
    graphviz.Source
        A renderable Graphviz source object. Call ``.render()`` to save,
        or display in Jupyter with ``IPython.display.SVG(source.pipe())``.

    Raises
    ------
    ImportError
        If ``graphviz`` is not installed.

    Examples
    --------
    >>> from deepbullwhip.chain.config import beer_game_config
    >>> from deepbullwhip.chain.graph import from_serial
    >>> graph = from_serial(beer_game_config())
    >>> source = render_network(graph, engine="dot", rankdir="TB")
    >>> source.render("output", format="pdf", cleanup=True)  # doctest: +SKIP
    """
    gv = import_optional("graphviz", "viz")

    # Build result lookup
    result_map: dict[str, Any] = {}
    if sim_result is not None:
        if isinstance(sim_result, NetworkSimulationResult):
            result_map = sim_result.node_results
        else:
            # SimulationResult: map by echelon index
            topo = graph.topological_order()
            demand_first = list(reversed(topo))
            for i, er in enumerate(sim_result.echelon_results):
                if i < len(demand_first):
                    result_map[demand_first[i]] = er

    dot_lines = [
        "digraph SupplyChain {",
        f'    rankdir={rankdir};',
        '    bgcolor="white";',
        '    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '    edge [fontname="Helvetica", fontsize=8, color="#888888"];',
    ]

    if title:
        dot_lines.append('    labelloc="t";')
        dot_lines.append(f'    label="{title}";')
        dot_lines.append('    fontsize=14;')
        dot_lines.append('    fontname="Helvetica Bold";')

    # Nodes
    for i, (name, cfg) in enumerate(graph.nodes.items()):
        label_parts = [f"<b>{name}</b>"]
        label_parts.append(f"LT={cfg.lead_time} | h={cfg.holding_cost:.2f} | b={cfg.backorder_cost:.2f}")

        color = _echelon_color(i)
        fontcolor = "white"

        if name in result_map:
            er = result_map[name]
            color = _bw_color(er.bullwhip_ratio)
            label_parts.append(
                f"BW={er.bullwhip_ratio:.2f} | FR={er.fill_rate:.0%}"
            )
            label_parts.append(f"Cost={er.total_cost:,.0f}")

        label = "<" + "<br/>".join(label_parts) + ">"
        dot_lines.append(
            f'    "{name}" [label={label}, fillcolor="{color}", '
            f'fontcolor="{fontcolor}"];'
        )

    # Edges
    for (upstream, downstream), edge_cfg in graph.edges.items():
        edge_label = f"LT={edge_cfg.lead_time}"
        if edge_cfg.capacity < float("inf"):
            edge_label += f"\\ncap={edge_cfg.capacity:.0f}"
        if edge_cfg.transport_cost > 0:
            edge_label += f"\\ntc={edge_cfg.transport_cost:.2f}"

        dot_lines.append(
            f'    "{upstream}" -> "{downstream}" '
            f'[label="{edge_label}", arrowsize=0.8];'
        )

    dot_lines.append("}")

    source_text = "\n".join(dot_lines)
    return gv.Source(source_text, engine=engine, format=fmt)


def render_simulation_snapshot(
    graph: SupplyChainGraph,
    sim_result: NetworkSimulationResult,
    period: int,
    engine: str = "dot",
    fmt: str = "svg",
    rankdir: str = "LR",
) -> Any:
    """Render the network state at a specific simulation period.

    Shows per-node inventory levels and per-edge order quantities
    at time step *period*.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology.
    sim_result : NetworkSimulationResult
        Simulation results containing time series data.
    period : int
        The time period (0-indexed) to visualize.
    engine : str
        Graphviz layout engine (default ``"dot"``).
    fmt : str
        Output format (default ``"svg"``).
    rankdir : str
        Layout direction (default ``"LR"``).

    Returns
    -------
    graphviz.Source
        A renderable Graphviz source object.

    Raises
    ------
    ImportError
        If ``graphviz`` is not installed.
    IndexError
        If *period* is out of range.

    Examples
    --------
    >>> # After running a simulation:
    >>> source = render_simulation_snapshot(graph, result, period=10)
    >>> source.render("snapshot_t10", format="svg")  # doctest: +SKIP
    """
    gv = import_optional("graphviz", "viz")

    dot_lines = [
        "digraph SupplyChain {",
        f'    rankdir={rankdir};',
        '    bgcolor="white";',
        '    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '    edge [fontname="Helvetica", fontsize=8];',
        '    labelloc="t";',
        f'    label="Supply Chain State at t={period}";',
        '    fontsize=14;',
        '    fontname="Helvetica Bold";',
    ]

    # Nodes with inventory snapshot
    for i, name in enumerate(graph.nodes):
        er = sim_result.node_results[name]
        inv = er.inventory_levels[period]
        order = er.orders[period]
        cost = er.costs[period]

        # Color by inventory status
        if inv >= 0:
            color = "#2E8B57"  # green: in stock
        else:
            color = "#CD5C5C"  # red: backorder

        label_parts = [
            f"<b>{name}</b>",
            f"Inv={inv:.1f}",
            f"Order={order:.1f}",
            f"Cost={cost:.1f}",
        ]
        label = "<" + "<br/>".join(label_parts) + ">"

        dot_lines.append(
            f'    "{name}" [label={label}, fillcolor="{color}", '
            f'fontcolor="white"];'
        )

    # Edges with flow quantities
    for (upstream, downstream), flows in sim_result.edge_flows.items():
        flow = flows[period] if period < len(flows) else 0.0
        edge_cfg = graph.edges.get((upstream, downstream))
        lt = edge_cfg.lead_time if edge_cfg else "?"

        color = "#006747" if flow > 0 else "#CCCCCC"
        penwidth = str(max(0.5, min(3.0, flow / 5.0)))

        dot_lines.append(
            f'    "{upstream}" -> "{downstream}" '
            f'[label="flow={flow:.1f}\\nLT={lt}", '
            f'color="{color}", penwidth={penwidth}, arrowsize=0.8];'
        )

    dot_lines.append("}")

    source_text = "\n".join(dot_lines)
    return gv.Source(source_text, engine=engine, format=fmt)


def save_figure(source: Any, filepath: str) -> str:
    """Save a Graphviz source to a file.

    The format is inferred from the file extension. Supported:
    ``.svg``, ``.pdf``, ``.png``.

    Parameters
    ----------
    source : graphviz.Source
        A Graphviz source object (from :func:`render_network` or
        :func:`render_simulation_snapshot`).
    filepath : str
        Output file path. The extension determines the format.

    Returns
    -------
    str
        The path of the rendered file.

    Examples
    --------
    >>> source = render_network(graph)
    >>> save_figure(source, "network.pdf")  # doctest: +SKIP
    """
    import os

    ext = os.path.splitext(filepath)[1].lstrip(".")
    base = os.path.splitext(filepath)[0]

    if ext not in ("svg", "pdf", "png"):
        raise ValueError(
            f"Unsupported format '{ext}'. Use 'svg', 'pdf', or 'png'."
        )

    return source.render(base, format=ext, cleanup=True)
