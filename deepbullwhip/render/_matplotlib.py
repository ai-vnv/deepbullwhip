"""Matplotlib backend for supply chain network rendering.

Produces publication-quality diagrams with solid colored rounded-rect
nodes, white bold labels, and clean directed edges with lead-time
labels. Inspired by D3Trees.jl and Bayesian network conventions.

Nodes show the echelon **name** by default. When ``sim_result`` is
provided, nodes are color-coded by bullwhip severity and display
BW/FR metrics.
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


def _node_box_size(
    name: str,
    theme: Theme,
    has_metrics: bool,
) -> tuple[float, float]:
    """Compute node rectangle size to fit text content."""
    char_w = theme.font.node_label_size * 0.011
    w = max(theme.node.min_width, len(name) * char_w + 0.35)
    if has_metrics:
        # Metrics line may be wider
        metrics_len = 20  # "BW=1.23 | FR=95%"
        w = max(w, metrics_len * char_w * 0.85 + 0.3)
    h = theme.node.min_height
    if has_metrics:
        h += 0.2
    return w, h


def render_matplotlib(
    graph: SupplyChainGraph,
    positions: dict[str, tuple[float, float]],
    theme: Theme,
    sim_result: SimulationResult | NetworkSimulationResult | None = None,
    title: str | None = None,
    annotations: dict[str, dict[str, str]] | None = None,
) -> matplotlib.figure.Figure:
    """Render a supply chain network using matplotlib.

    Without ``sim_result``: clean diagram with node names and edge
    lead times. With ``sim_result``: nodes color-coded by bullwhip
    severity (green/gold/red) with BW and fill rate metrics.

    Parameters
    ----------
    graph : SupplyChainGraph
        The supply chain topology.
    positions : dict[str, tuple[float, float]]
        Node positions.
    theme : Theme
        Visual theme controlling all styling.
    sim_result : SimulationResult or NetworkSimulationResult or None
        Optional simulation results. Adds BW/FR metrics to nodes and
        color-codes them by bullwhip severity.
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

    # Pre-compute node sizes
    node_sizes: dict[str, tuple[float, float]] = {}
    for name in graph.nodes:
        has_metrics = name in result_map
        node_sizes[name] = _node_box_size(name, theme, has_metrics)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    margin = theme.figure.margin * 2
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

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
        w0, h0 = node_sizes[upstream]
        w1, h1 = node_sizes[downstream]
        clip0 = np.sqrt((w0 / 2) ** 2 + (h0 / 2) ** 2) * 0.72
        clip1 = np.sqrt((w1 / 2) ** 2 + (h1 / 2) ** 2) * 0.72
        sx, sy = x0 + ux * clip0, y0 + uy * clip0
        ex, ey = x1 - ux * clip1, y1 - uy * clip1

        ax.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=theme.edge.color,
                lw=theme.edge.line_width,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=8,
            ),
            zorder=2,
        )

        # Edge label -- offset perpendicular (or above for horizontal)
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        offset = 0.2
        if abs(uy) < 0.3:
            # Nearly horizontal: push label above
            px, py = 0, offset
        else:
            px, py = -uy * offset, ux * offset
        ax.text(
            mx + px,
            my + py,
            f"LT={edge_cfg.lead_time}",
            ha="center",
            va="center",
            fontsize=theme.font.edge_label_size,
            fontfamily=theme.font.family,
            color="#666666",
            zorder=3,
        )

    # --- Draw nodes ---
    node_names = list(graph.nodes.keys())

    for i, name in enumerate(node_names):
        if name not in positions:
            continue

        x, y = positions[name]
        w, h = node_sizes[name]
        has_metrics = name in result_map

        # Color: BW-severity when sim_result, else theme palette
        if has_metrics:
            color = theme.bw_color(result_map[name].bullwhip_ratio)
        else:
            color = theme.node_color(i)

        # Rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad={theme.node.corner_radius}",
            facecolor=color,
            edgecolor="white",
            linewidth=theme.node.border_width * 1.5,
            zorder=5,
        )
        ax.add_patch(rect)

        # --- Text inside node ---
        if has_metrics:
            # Two lines: name + metrics
            er = result_map[name]
            ax.text(
                x,
                y + 0.1,
                name,
                ha="center",
                va="center",
                fontsize=theme.font.node_label_size,
                fontweight="bold",
                fontfamily=theme.font.family,
                color="white",
                zorder=6,
            )
            ax.text(
                x,
                y - 0.12,
                f"BW={er.bullwhip_ratio:.2f} | FR={er.fill_rate:.0%}",
                ha="center",
                va="center",
                fontsize=theme.font.node_detail_size,
                fontweight="bold",
                fontfamily=theme.font.family,
                color=(1, 1, 1, 0.9),
                zorder=6,
            )
        else:
            # Single line: name only
            ax.text(
                x,
                y,
                name,
                ha="center",
                va="center",
                fontsize=theme.font.node_label_size,
                fontweight="bold",
                fontfamily=theme.font.family,
                color="white",
                zorder=6,
            )

        # Extra annotations (outside, below node)
        if annotations and name in annotations:
            parts = [f"{k}={v}" for k, v in annotations[name].items()]
            ax.text(
                x,
                y - h / 2 - 0.12,
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
