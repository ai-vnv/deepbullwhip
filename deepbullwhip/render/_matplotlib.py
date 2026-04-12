"""Matplotlib backend for supply chain network rendering.

Produces publication-quality network diagrams with solid colored
circle nodes, white bold labels, and directed edges with numeric
labels placed directly on the edge lines.
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


def _build_result_map(
    graph: SupplyChainGraph,
    sim_result: SimulationResult | NetworkSimulationResult | None,
) -> dict[str, Any]:
    """Build a name -> EchelonResult lookup from either result type."""
    if sim_result is None:
        return {}
    if isinstance(sim_result, NetworkSimulationResult):
        return dict(sim_result.node_results)
    topo = graph.topological_order()
    demand_first = list(reversed(topo))
    result_map = {}
    for i, er in enumerate(sim_result.echelon_results):
        if i < len(demand_first):
            result_map[demand_first[i]] = er
    return result_map


def _auto_fontsize(name: str, radius: float) -> float:
    """Pick font size so text fits inside the circle."""
    max_width = radius * 1.4  # usable text width inside circle
    char_width = 0.085  # approximate character width at size 10
    ideal = max_width / (len(name) * char_width) * 10
    return min(10.0, max(5.5, ideal))


def render_matplotlib(
    graph: SupplyChainGraph,
    positions: dict[str, tuple[float, float]],
    theme: Theme,
    sim_result: SimulationResult | NetworkSimulationResult | None = None,
    title: str | None = None,
    annotations: dict[str, dict[str, str]] | None = None,
) -> matplotlib.figure.Figure:
    """Render a supply chain network using matplotlib.

    Draws solid colored circle nodes with white bold labels and
    directed edges with numeric lead-time labels placed directly
    on the edge lines.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology.
    positions : dict[str, tuple[float, float]]
        Node positions.
    theme : Theme
        Visual theme controlling all styling.
    sim_result : SimulationResult or NetworkSimulationResult or None
        Optional simulation results. Adds BW/FR annotation below
        each node and color-codes by bullwhip severity.
    title : str or None
        Figure title.
    annotations : dict[str, dict[str, str]] or None
        Extra per-node text annotations (shown below node).

    Returns
    -------
    matplotlib.figure.Figure
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

    # Node radius -- keep large enough for readable text always
    R = theme.node.min_width / 2.0

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pad = R + 0.6
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    # --- Draw edges ---
    for (upstream, downstream), edge_cfg in graph.edges.items():
        if upstream not in positions or downstream not in positions:
            continue

        x0, y0 = positions[upstream]
        x1, y1 = positions[downstream]
        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            continue

        ux, uy = dx / dist, dy / dist
        sx, sy = x0 + ux * (R + 0.05), y0 + uy * (R + 0.05)
        ex, ey = x1 - ux * (R + 0.05), y1 - uy * (R + 0.05)

        ax.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="->",
                color=theme.edge.color,
                lw=theme.edge.line_width,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=10,
            ),
            zorder=2,
        )

        # Label ON the edge (midpoint with white background)
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.text(
            mx,
            my,
            str(edge_cfg.lead_time),
            ha="center",
            va="center",
            fontsize=theme.font.edge_label_size + 1,
            fontweight="bold",
            fontfamily=theme.font.family,
            color=theme.edge.color,
            bbox=dict(
                boxstyle="round,pad=0.12",
                fc=theme.figure.background,
                ec="none",
            ),
            zorder=3,
        )

    # --- Draw nodes ---
    node_names = list(graph.nodes.keys())

    for i, name in enumerate(node_names):
        if name not in positions:
            continue

        x, y = positions[name]

        # Color
        if name in result_map:
            color = theme.bw_color(result_map[name].bullwhip_ratio)
        else:
            color = theme.node_color(i)

        # Circle
        circle = mpatches.Circle(
            (x, y),
            R,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
            zorder=5,
        )
        ax.add_patch(circle)

        # Node name (white bold, auto-sized)
        fs = _auto_fontsize(name, R)
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold",
            fontfamily=theme.font.family,
            color="white",
            zorder=6,
        )

        # Metrics below node (if sim_result)
        if name in result_map:
            er = result_map[name]
            ax.text(
                x,
                y - R - 0.15,
                f"BW={er.bullwhip_ratio:.2f}  FR={er.fill_rate:.0%}",
                ha="center",
                va="top",
                fontsize=theme.font.node_detail_size,
                fontfamily=theme.font.family,
                color="#666666",
                zorder=6,
            )

        # Extra annotations below
        if annotations and name in annotations:
            ann_y = y - R - 0.15
            if name in result_map:
                ann_y -= 0.25
            parts = [f"{k}={v}" for k, v in annotations[name].items()]
            ax.text(
                x,
                ann_y,
                "  ".join(parts),
                ha="center",
                va="top",
                fontsize=theme.font.node_detail_size,
                fontfamily=theme.font.family,
                color="#888888",
                zorder=6,
            )

    # Title
    if title:
        ax.set_title(
            title,
            fontsize=theme.font.title_size,
            fontweight=theme.font.title_weight,
            fontfamily=theme.font.family,
            pad=12,
            color="#333333",
        )

    fig.tight_layout()
    return fig
