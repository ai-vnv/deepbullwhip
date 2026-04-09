"""Supply chain network diagram and geographic map visualizations.

Provides two publication-grade plot types:
1. Abstract network diagram (no external dependencies beyond matplotlib)
2. Geographic map visualization with node locations on a coordinate system
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import matplotlib
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from deepbullwhip._types import SimulationResult
from deepbullwhip.diagnostics.plots import (
    COLORS,
    DOUBLE_COL,
    GOLDEN,
    SINGLE_COL,
    _apply_style,
    _echelon_color,
)


@dataclass
class NodeLocation:
    """Geographic or schematic location of a supply chain node."""

    name: str
    lat: float
    lon: float
    role: str = ""
    details: str = ""


@dataclass
class SupplyChainNetwork:
    """Describes the topology and geography of a supply chain."""

    nodes: list[NodeLocation]
    edges: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.edges and len(self.nodes) > 1:
            # Default: serial chain E_K -> ... -> E_1
            self.edges = [(i + 1, i) for i in range(len(self.nodes) - 1)]


def kfupm_petrochemical_network() -> SupplyChainNetwork:
    """Example 4-echelon petrochemical supply chain in Saudi Arabia.

    A petrochemical product (polyethylene) supply chain involving KFUPM
    as R&D partner, sourced from Eastern Province refineries, processed
    through Jubail industrial complex, and distributed domestically.
    """
    nodes = [
        NodeLocation(
            name="KFUPM / Distributor",
            lat=26.3073, lon=50.1433,
            role="Distributor / OEM",
            details="KFUPM Dhahran campus\nR&D + regional distribution hub",
        ),
        NodeLocation(
            name="Jubail OSAT Plant",
            lat=27.0046, lon=49.6588,
            role="Assembly & Processing",
            details="Jubail Industrial City\nPolymer compounding & packaging",
        ),
        NodeLocation(
            name="SABIC / Yanbu Refinery",
            lat=24.0895, lon=38.0618,
            role="Foundry / Manufacturing",
            details="Yanbu Industrial City\nPetrochemical production (SABIC)",
        ),
        NodeLocation(
            name="Aramco Raw Materials",
            lat=25.3838, lon=49.9164,
            role="Raw Material Supplier",
            details="Abqaiq / Ras Tanura\nCrude oil & naphtha feedstock",
        ),
    ]
    return SupplyChainNetwork(nodes=nodes)


# ── Network diagram ─────────────────────────────────────────────────


def plot_network_diagram(
    network: SupplyChainNetwork,
    sim_result: SimulationResult | None = None,
    width: Literal["single", "double"] = "double",
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> matplotlib.figure.Figure:
    """Abstract network diagram of the supply chain.

    Nodes are drawn as rounded rectangles with role labels. Edges show
    material flow direction. If sim_result is provided, node annotations
    include BW ratio and fill rate.
    """
    _apply_style()
    w = SINGLE_COL if width == "single" else DOUBLE_COL
    K = len(network.nodes)

    if orientation == "horizontal":
        fig, ax = plt.subplots(figsize=(w, w / GOLDEN / 1.8))
        positions = [(i * 1.0 / (K - 1) if K > 1 else 0.5, 0.5) for i in range(K)]
    else:
        fig, ax = plt.subplots(figsize=(w / 1.5, w * 0.9))
        positions = [(0.5, 1.0 - i * 1.0 / (K - 1) if K > 1 else 0.5) for i in range(K)]

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    node_w = 0.18
    node_h = 0.22 if orientation == "horizontal" else 0.12

    # Draw edges first (behind nodes)
    for src, dst in network.edges:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color="#888888", lw=1.2,
                connectionstyle="arc3,rad=0.0",
            ),
        )

    # Draw nodes
    for k, (node, (x, y)) in enumerate(zip(network.nodes, positions)):
        color = _echelon_color(k)
        rect = mpatches.FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="black", linewidth=0.6, alpha=0.25,
        )
        ax.add_patch(rect)

        # Node label
        label_lines = [f"E{k + 1}: {node.role}"]
        if node.name:
            label_lines.insert(0, node.name)

        if sim_result is not None and k < len(sim_result.echelon_results):
            er = sim_result.echelon_results[k]
            label_lines.append(f"BW={er.bullwhip_ratio:.2f}  FR={er.fill_rate:.0%}")

        ax.text(
            x, y, "\n".join(label_lines),
            ha="center", va="center", fontsize=6,
            fontweight="bold" if k == 0 else "normal",
            linespacing=1.4,
        )

    # Flow label
    if orientation == "horizontal":
        ax.text(0.5, -0.02, r"Material flow $\longrightarrow$",
                ha="center", va="top", fontsize=7, color="#666666")
    else:
        ax.text(0.85, 0.5, r"Material flow $\downarrow$",
                ha="left", va="center", fontsize=7, color="#666666", rotation=90)

    return fig


# ── Geographic map ───────────────────────────────────────────────────


def plot_supply_chain_map(
    network: SupplyChainNetwork,
    sim_result: SimulationResult | None = None,
    width: Literal["single", "double"] = "double",
    map_bounds: tuple[float, float, float, float] | None = None,
    show_country_outline: bool = True,
) -> matplotlib.figure.Figure:
    """Geographic visualization of supply chain nodes on a lat/lon plot.

    Plots nodes at their geographic coordinates with connecting arcs.
    If sim_result is provided, node size scales with total cost and
    color intensity with bullwhip ratio.

    Parameters
    ----------
    map_bounds : (lat_min, lat_max, lon_min, lon_max) or None
        If None, computed from node positions with padding.
    show_country_outline : bool
        If True, draws a simplified Saudi Arabia outline.
    """
    _apply_style()
    w = SINGLE_COL if width == "single" else DOUBLE_COL
    fig, ax = plt.subplots(figsize=(w, w * 0.85))

    lats = [n.lat for n in network.nodes]
    lons = [n.lon for n in network.nodes]

    if map_bounds is None:
        pad_lat = max(1.5, (max(lats) - min(lats)) * 0.25)
        pad_lon = max(1.5, (max(lons) - min(lons)) * 0.25)
        map_bounds = (
            min(lats) - pad_lat, max(lats) + pad_lat,
            min(lons) - pad_lon, max(lons) + pad_lon,
        )

    if show_country_outline:
        _draw_saudi_outline(ax)

    # Draw edges (arcs)
    for src, dst in network.edges:
        n_src, n_dst = network.nodes[src], network.nodes[dst]
        ax.annotate(
            "", xy=(n_dst.lon, n_dst.lat), xytext=(n_src.lon, n_src.lat),
            arrowprops=dict(
                arrowstyle="-|>", color="#888888", lw=0.8,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=2,
        )

    # Draw nodes
    for k, node in enumerate(network.nodes):
        color = _echelon_color(k)
        size = 80
        alpha = 0.85

        if sim_result is not None and k < len(sim_result.echelon_results):
            er = sim_result.echelon_results[k]
            size = 60 + 40 * min(er.bullwhip_ratio, 5)
            alpha = 0.7 + 0.06 * min(er.bullwhip_ratio, 5)

        ax.scatter(
            node.lon, node.lat, s=size, c=color, edgecolors="black",
            linewidths=0.5, alpha=alpha, zorder=5,
        )

        # Label
        label = f"E{k + 1}: {node.name}"
        if sim_result is not None and k < len(sim_result.echelon_results):
            er = sim_result.echelon_results[k]
            label += f"\nBW={er.bullwhip_ratio:.2f}"

        offset_y = 0.4
        ax.annotate(
            label, (node.lon, node.lat),
            textcoords="offset points", xytext=(8, 8),
            fontsize=5.5, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85, lw=0.4),
            zorder=6,
        )

    ax.set_xlim(map_bounds[2], map_bounds[3])
    ax.set_ylim(map_bounds[0], map_bounds[1])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(lats))))
    ax.grid(True, alpha=0.15, linewidth=0.3)

    return fig


def _draw_saudi_outline(ax: plt.Axes) -> None:
    """Draw a simplified Saudi Arabia coastline/border polygon.

    This is a coarse approximation for visual context only.
    Points are (lon, lat) ordered clockwise.
    """
    # Simplified Saudi Arabia boundary (approx 20 vertices)
    outline_lon = [
        36.5, 37.5, 39.2, 40.0, 41.5, 42.0, 43.0, 43.5,
        45.0, 47.0, 49.5, 50.8, 51.6, 51.2, 50.5, 50.0,
        49.0, 48.5, 48.0, 47.5, 47.0, 46.5, 44.5, 42.0,
        39.5, 38.0, 36.5, 35.0, 34.5, 36.5,
    ]
    outline_lat = [
        29.0, 27.5, 26.0, 24.5, 23.5, 22.0, 20.5, 19.0,
        17.5, 16.5, 16.5, 18.0, 19.5, 22.0, 22.5, 23.5,
        24.0, 24.5, 25.0, 25.5, 26.0, 27.0, 27.5, 28.0,
        28.5, 29.5, 30.0, 30.5, 29.5, 29.0,
    ]
    ax.fill(outline_lon, outline_lat, color="#F5F0E8", edgecolor="#CCCCCC",
            linewidth=0.5, zorder=0)
    ax.plot(outline_lon, outline_lat, color="#BBBBBB", linewidth=0.4, zorder=1)
