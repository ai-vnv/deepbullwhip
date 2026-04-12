"""TikZ/LaTeX backend for supply chain network rendering.

Produces a complete ``.tex`` source string containing a TikZ picture
of the supply chain network. No LaTeX compilation is performed --
the output is a plain string that can be written to a file and
compiled with ``pdflatex`` or included in a LaTeX document.

Uses ``jinja2`` (already a core dependency) for template rendering.

The output uses standard TikZ libraries:

- ``arrows.meta`` for arrow styles
- ``positioning`` for relative node placement
- ``calc`` for coordinate arithmetic

Examples
--------
>>> from deepbullwhip.chain.config import beer_game_config
>>> from deepbullwhip.chain.graph import from_serial
>>> from deepbullwhip.render._tikz import render_tikz
>>> from deepbullwhip.render.layout import compute_positions
>>> from deepbullwhip.render.theme import get_theme
>>>
>>> graph = from_serial(beer_game_config())
>>> positions = compute_positions(graph)
>>> tex = render_tikz(graph, positions, get_theme("kfupm"))
>>> "\\\\begin{tikzpicture}" in tex
True
"""

from __future__ import annotations


import jinja2

from deepbullwhip._types import NetworkSimulationResult, SimulationResult
from deepbullwhip.chain.graph import SupplyChainGraph
from deepbullwhip.render._matplotlib import _build_result_map
from deepbullwhip.render.theme import Theme

_TIKZ_TEMPLATE = r"""
{%- if standalone %}
\documentclass[border=5mm]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, calc}
{% for name, hex in colors.items() -%}
\definecolor{{ '{' }}{{ name }}{{ '}' }}{HTML}{{ '{' }}{{ hex }}{{ '}' }}
{% endfor -%}
\begin{document}
{% endif -%}
\begin{tikzpicture}[
    every node/.style={
        draw,
        rounded corners={{ corner_radius }}pt,
        minimum width={{ min_width }}cm,
        minimum height={{ min_height }}cm,
        align=center,
        font={{ font_cmd }}\fontsize{{ '{' }}{{ label_size }}{{ '}' }}{{ '{' }}{{ detail_size }}{{ '}' }}\selectfont,
        line width={{ border_width }}pt,
    },
    every edge/.style={
        draw,
        {{ edge_color }},
        line width={{ edge_width }}pt,
        -{Stealth[length=3pt, width=2.5pt]},
    },
]

{% for node in nodes -%}
\node[fill={{ node.color }}!{{ fill_pct }}] ({{ node.id }}) at ({{ node.x }}cm, {{ node.y }}cm) {
    \textbf{{ '{' }}{{ node.label }}{{ '}' }}\\[1pt]
    {\fontsize{{ '{' }}{{ detail_size }}{{ '}' }}{{ '{' }}{{ detail_size }}{{ '}' }}\selectfont {{ node.detail }}}
};
{% endfor %}
{% for edge in edges -%}
\draw[{{ edge_color }}] ({{ edge.source }}) -- node[draw=none, fill=white, fill opacity=0.8, text opacity=1, font=\fontsize{{ '{' }}{{ edge_label_size }}{{ '}' }}{{ '{' }}{{ edge_label_size }}{{ '}' }}\selectfont, minimum width=0cm, minimum height=0cm, inner sep=1pt] {{ '{' }}{{ edge.label }}{{ '}' }} ({{ edge.target }});
{% endfor -%}
{% if title -%}
\node[draw=none, fill=none, minimum width=0cm, minimum height=0cm, font=\bfseries\fontsize{{ '{' }}{{ title_size }}{{ '}' }}{{ '{' }}{{ title_size }}{{ '}' }}\selectfont] at ({{ title_x }}cm, {{ title_y }}cm) {{ '{' }}{{ title }}{{ '}' }};
{% endif -%}

\end{tikzpicture}
{%- if standalone %}
\end{document}
{%- endif %}
"""


def _hex_to_tikz_name(hex_color: str) -> str:
    """Convert a hex color to a sanitized TikZ color name."""
    return "clr" + hex_color.lstrip("#").upper()


def _font_command(family: str) -> str:
    """Map font family to a LaTeX font command."""
    if family == "sans-serif":
        return r"\sffamily"
    return r"\rmfamily"


def render_tikz(
    graph: SupplyChainGraph,
    positions: dict[str, tuple[float, float]],
    theme: Theme,
    sim_result: SimulationResult | NetworkSimulationResult | None = None,
    title: str | None = None,
    standalone: bool = True,
) -> str:
    """Render a supply chain network as TikZ/LaTeX source code.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology.
    positions : dict[str, tuple[float, float]]
        Node positions in ``(x, y)`` centimeters.
    theme : Theme
        Visual theme.
    sim_result : SimulationResult or NetworkSimulationResult or None
        Optional simulation results.
    title : str or None
        Figure title.
    standalone : bool
        If ``True``, wrap in ``\\documentclass{standalone}``.
        If ``False``, output only the ``tikzpicture`` environment.

    Returns
    -------
    str
        Complete LaTeX/TikZ source code.
    """
    result_map = _build_result_map(graph, sim_result)

    # Collect unique colors
    color_defs: dict[str, str] = {}
    edge_color_name = _hex_to_tikz_name(theme.edge.color)
    color_defs[edge_color_name] = theme.edge.color.lstrip("#")

    # Build node data
    node_data = []
    for i, (name, cfg) in enumerate(graph.nodes.items()):
        if name in result_map:
            color_hex = theme.bw_color(result_map[name].bullwhip_ratio)
        else:
            color_hex = theme.node_color(i)

        color_name = _hex_to_tikz_name(color_hex)
        color_defs[color_name] = color_hex.lstrip("#")

        # Escape LaTeX special chars
        safe_name = name.replace("_", r"\_")
        safe_id = name.replace(" ", "").replace("_", "")

        detail_parts = [
            f"LT={cfg.lead_time}  h={cfg.holding_cost:.2f}  b={cfg.backorder_cost:.2f}"
        ]
        if name in result_map:
            er = result_map[name]
            detail_parts.append(
                f"BW={er.bullwhip_ratio:.2f}  FR={er.fill_rate:.0%}"
            )

        x, y = positions.get(name, (0, 0))

        node_data.append({
            "id": safe_id,
            "label": safe_name,
            "detail": " \\\\\\ ".join(detail_parts),
            "color": color_name,
            "x": f"{x:.1f}",
            "y": f"{y:.1f}",
        })

    # Build edge data
    edge_data = []
    for (upstream, downstream), edge_cfg in graph.edges.items():
        safe_src = upstream.replace(" ", "").replace("_", "")
        safe_tgt = downstream.replace(" ", "").replace("_", "")
        label = f"LT={edge_cfg.lead_time}"

        edge_data.append({
            "source": safe_src,
            "target": safe_tgt,
            "label": label,
        })

    # Title position
    if positions:
        ys = [p[1] for p in positions.values()]
        title_x = sum(p[0] for p in positions.values()) / len(positions)
        title_y = max(ys) + 1.5
    else:
        title_x, title_y = 0.0, 2.0

    fill_pct = int(theme.node.fill_alpha * 100)

    template = jinja2.Template(_TIKZ_TEMPLATE)
    return template.render(
        standalone=standalone,
        colors=color_defs,
        corner_radius=int(theme.node.corner_radius * 100),
        min_width=theme.node.min_width,
        min_height=theme.node.min_height,
        border_width=theme.node.border_width,
        font_cmd=_font_command(theme.font.family),
        label_size=int(theme.font.node_label_size),
        detail_size=int(theme.font.node_detail_size),
        edge_color=edge_color_name,
        edge_width=theme.edge.line_width,
        edge_label_size=int(theme.font.edge_label_size),
        fill_pct=fill_pct,
        nodes=node_data,
        edges=edge_data,
        title=title,
        title_size=int(theme.font.title_size),
        title_x=f"{title_x:.1f}",
        title_y=f"{title_y:.1f}",
    )
