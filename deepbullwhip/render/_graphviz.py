"""Graphviz backend for supply chain network rendering.

Produces DOT-language graphs rendered via the ``graphviz`` Python
package. Supports multiple layout engines (``dot``, ``neato``, ``fdp``)
and output formats (SVG, PDF, PNG).

Requires the ``graphviz`` optional dependency::

    pip install deepbullwhip[viz]
"""

from __future__ import annotations

from typing import Any

from deepbullwhip._optional import import_optional
from deepbullwhip._types import NetworkSimulationResult, SimulationResult
from deepbullwhip.chain.graph import SupplyChainGraph
from deepbullwhip.render._matplotlib import _build_result_map
from deepbullwhip.render.theme import Theme


def _font_name(family: str) -> str:
    """Map font family to a Graphviz font name."""
    if family == "serif":
        return "Times"
    elif family == "sans-serif":
        return "Helvetica"
    return family


def render_graphviz(
    graph: SupplyChainGraph,
    positions: dict[str, tuple[float, float]] | None,
    theme: Theme,
    sim_result: SimulationResult | NetworkSimulationResult | None = None,
    title: str | None = None,
    engine: str = "dot",
    fmt: str = "svg",
) -> Any:
    """Render a supply chain network using Graphviz DOT.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology.
    positions : dict[str, tuple[float, float]] or None
        Node positions. If provided with ``engine="neato"``, nodes
        are pinned to these positions.
    theme : Theme
        Visual theme.
    sim_result : SimulationResult or NetworkSimulationResult or None
        Optional simulation results for metric overlay.
    title : str or None
        Graph title.
    engine : str
        Graphviz layout engine: ``"dot"``, ``"neato"``, ``"fdp"``.
    fmt : str
        Output format: ``"svg"``, ``"pdf"``, ``"png"``.

    Returns
    -------
    graphviz.Source
        A renderable Graphviz source object.

    Raises
    ------
    ImportError
        If ``graphviz`` is not installed.
    """
    gv = import_optional("graphviz", "viz")
    result_map = _build_result_map(graph, sim_result)
    fontname = _font_name(theme.font.family)

    # Use neato with pinned positions if explicit positions provided
    use_positions = positions is not None and engine in ("neato", "fdp")

    lines = [
        "digraph SupplyChain {",
        "    rankdir=TB;",
        f'    bgcolor="{theme.figure.background}";',
        f'    node [shape=box, style="rounded,filled", fontname="{fontname}", '
        f'fontsize={theme.font.node_label_size}];',
        f'    edge [fontname="{fontname}", fontsize={theme.font.edge_label_size}, '
        f'color="{theme.edge.color}"];',
    ]

    if title:
        lines.append('    labelloc="t";')
        lines.append(f'    label="{title}";')
        lines.append(f"    fontsize={theme.font.title_size};")
        lines.append(f'    fontname="{fontname}";')

    # Nodes
    for i, (name, cfg) in enumerate(graph.nodes.items()):
        label_parts = [f"<b>{name}</b>"]
        label_parts.append(
            f"LT={cfg.lead_time} | h={cfg.holding_cost:.2f} | b={cfg.backorder_cost:.2f}"
        )

        if name in result_map:
            er = result_map[name]
            color = theme.bw_color(er.bullwhip_ratio)
            label_parts.append(f"BW={er.bullwhip_ratio:.2f} | FR={er.fill_rate:.0%}")
        else:
            color = theme.node_color(i)

        label = "<" + "<br/>".join(label_parts) + ">"

        attrs = [
            f'label={label}',
            f'fillcolor="{color}{int(theme.node.fill_alpha * 255):02x}"',
            'fontcolor="white"',
            f'penwidth={theme.node.border_width}',
        ]

        if use_positions and name in positions:
            x, y = positions[name]
            attrs.append(f'pos="{x},{y}!"')

        lines.append(f'    "{name}" [{", ".join(attrs)}];')

    # Edges
    for (upstream, downstream), edge_cfg in graph.edges.items():
        label = f"LT={edge_cfg.lead_time}"
        if edge_cfg.capacity < float("inf"):
            label += f"\\ncap={edge_cfg.capacity:.0f}"

        lines.append(
            f'    "{upstream}" -> "{downstream}" '
            f'[label="{label}", arrowsize=0.8, penwidth={theme.edge.line_width}];'
        )

    lines.append("}")

    source_text = "\n".join(lines)
    return gv.Source(source_text, engine=engine, format=fmt)
