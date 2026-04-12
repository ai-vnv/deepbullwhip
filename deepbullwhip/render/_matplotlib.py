"""Matplotlib backend for supply chain network rendering.

Produces publication-quality PNG/PDF figures using only matplotlib
(a core dependency). Draws Bayesian-network-style diagrams with
rounded rectangle nodes and directed edges.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from deepbullwhip._types import NetworkSimulationResult, SimulationResult
from deepbullwhip.chain.graph import SupplyChainGraph
from deepbullwhip.render.layout import compute_figure_size
from deepbullwhip.render.theme import Theme


def _hex_to_rgba(hex_color: str, alpha: float) -> tuple[float, ...]:
    """Convert hex color to RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def _build_result_map(
    graph: SupplyChainGraph,
    sim_result: SimulationResult | NetworkSimulationResult | None,
) -> dict[str, Any]:
    """Build a name -> EchelonResult lookup from either result type."""
    if sim_result is None:
        return {}
    if isinstance(sim_result, NetworkSimulationResult):
        return dict(sim_result.node_results)
    # SimulationResult: map by topological order (demand-first)
    topo = graph.topological_order()
    demand_first = list(reversed(topo))
    result_map = {}
    for i, er in enumerate(sim_result.echelon_results):
        if i < len(demand_first):
            result_map[demand_first[i]] = er
    return result_map


def render_matplotlib(
    graph: SupplyChainGraph,
    positions: dict[str, tuple[float, float]],
    theme: Theme,
    sim_result: SimulationResult | NetworkSimulationResult | None = None,
    title: str | None = None,
    annotations: dict[str, dict[str, str]] | None = None,
) -> matplotlib.figure.Figure:
    """Render a supply chain network using matplotlib.

    Produces a Bayesian-network-style diagram with rounded rectangle
    nodes, directed edges, and optional simulation metric overlays.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology.
    positions : dict[str, tuple[float, float]]
        Node positions (from :func:`~deepbullwhip.render.layout.compute_positions`).
    theme : Theme
        Visual theme controlling all styling.
    sim_result : SimulationResult or NetworkSimulationResult or None
        Optional simulation results for metric overlay.
    title : str or None
        Figure title.
    annotations : dict[str, dict[str, str]] or None
        Extra per-node text annotations.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure (call ``.savefig()`` to export).
    """
    result_map = _build_result_map(graph, sim_result)
    fig_size = compute_figure_size(positions, theme)

    fig, ax = plt.subplots(figsize=fig_size, dpi=theme.figure.dpi)
    ax.set_facecolor(theme.figure.background)
    fig.set_facecolor(theme.figure.background)
    ax.set_aspect("equal")
    ax.axis("off")

    if not positions:
        return fig

    # Compute data bounds for axis limits
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    margin = theme.figure.margin * 2
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    # Draw edges first (behind nodes)
    for (upstream, downstream), edge_cfg in graph.edges.items():
        if upstream in positions and downstream in positions:
            x0, y0 = positions[upstream]
            x1, y1 = positions[downstream]

            conn_style = "arc3,rad=0.0"
            if theme.edge.curve_radius > 0:
                conn_style = f"arc3,rad={theme.edge.curve_radius}"

            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=theme.edge.arrow_style,
                    color=theme.edge.color,
                    lw=theme.edge.line_width,
                    connectionstyle=conn_style,
                ),
                zorder=1,
            )

            # Edge label
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            label = f"LT={edge_cfg.lead_time}"
            ax.text(
                mid_x, mid_y, label,
                ha="center", va="center",
                fontsize=theme.font.edge_label_size,
                fontfamily=theme.font.family,
                color=theme.edge.color,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc=theme.figure.background,
                    ec="none",
                    alpha=0.8,
                ),
                zorder=2,
            )

    # Draw nodes
    node_names = list(graph.nodes.keys())
    node_w = theme.node.min_width
    node_h = theme.node.min_height

    for i, name in enumerate(node_names):
        if name not in positions:
            continue

        x, y = positions[name]
        cfg = graph.nodes[name]

        # Determine color
        if name in result_map:
            color = theme.bw_color(result_map[name].bullwhip_ratio)
        else:
            color = theme.node_color(i)

        # Draw rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2),
            node_w,
            node_h,
            boxstyle=f"round,pad={theme.node.corner_radius}",
            facecolor=_hex_to_rgba(color, theme.node.fill_alpha),
            edgecolor=color,
            linewidth=theme.node.border_width,
            zorder=3,
        )
        ax.add_patch(rect)

        # Build label text
        label_lines = [name]
        label_lines.append(
            f"LT={cfg.lead_time}  h={cfg.holding_cost:.2f}  b={cfg.backorder_cost:.2f}"
        )

        if name in result_map:
            er = result_map[name]
            label_lines.append(
                f"BW={er.bullwhip_ratio:.2f}  FR={er.fill_rate:.0%}"
            )

        if annotations and name in annotations:
            for key, val in annotations[name].items():
                label_lines.append(f"{key}={val}")

        # Node name (bold)
        ax.text(
            x, y + node_h * 0.15,
            label_lines[0],
            ha="center", va="center",
            fontsize=theme.font.node_label_size,
            fontfamily=theme.font.family,
            fontweight="bold",
            zorder=4,
        )

        # Node details
        detail_text = "\n".join(label_lines[1:])
        ax.text(
            x, y - node_h * 0.12,
            detail_text,
            ha="center", va="center",
            fontsize=theme.font.node_detail_size,
            fontfamily=theme.font.family,
            color="#333333",
            linespacing=1.4,
            zorder=4,
        )

    # Title
    if title:
        ax.set_title(
            title,
            fontsize=theme.font.title_size,
            fontweight=theme.font.title_weight,
            fontfamily=theme.font.family,
            pad=10,
        )

    fig.tight_layout()
    return fig
