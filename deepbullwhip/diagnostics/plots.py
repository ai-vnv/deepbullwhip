"""Publication-grade diagnostic plots for supply chain simulation.

All figures use consistent styling suitable for single-column (3.5 in)
or double-column (7.0 in) journal layouts. Fonts are serif-based for
LaTeX compatibility. Every function returns a Figure without calling
plt.show(), so the caller controls display and export.
"""

from __future__ import annotations

from typing import Literal

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from deepbullwhip._types import SimulationResult, TimeSeries

# ── Publication style defaults ──────────────────────────────────────────

SINGLE_COL = 3.5  # inches (IEEE / Elsevier single column)
DOUBLE_COL = 7.0  # inches (IEEE / Elsevier double column)
GOLDEN = (1 + np.sqrt(5)) / 2  # golden ratio ≈ 1.618

# Colour palette – KFUPM AI-VNV Lab (ai-vnv.kfupm.io)
COLORS = {
    "demand": "#1a1a1a",           # primary text
    "E1": "#006747",               # KFUPM green (primary)
    "E2": "#c4a35a",               # sand/gold (accent)
    "E3": "#004040",               # deep teal
    "E4": "#9e8340",               # dark sand
    "holding": "#006747",          # KFUPM green
    "backorder": "#c4a35a",        # sand/gold
    "highlight": "#004d35",        # dark green
    "grid": "#e5e7eb",             # border
    "bg_alt": "#fafafa",           # alternate background
    "primary_light": "#e8f5f0",    # light green tint
}
_ECHELON_COLORS = [COLORS["E1"], COLORS["E2"], COLORS["E3"], COLORS["E4"]]


def _apply_style() -> None:
    """Apply publication-grade matplotlib rcParams."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.grid": False,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def _echelon_color(k: int) -> str:
    return _ECHELON_COLORS[k % len(_ECHELON_COLORS)]


def _col_width(
    width: Literal["single", "double"] = "double",
) -> float:
    return SINGLE_COL if width == "single" else DOUBLE_COL


# ── 1. Demand trajectory ────────────────────────────────────────────────


def plot_demand_trajectory(
    demand: TimeSeries,
    shock_period: int | None = 104,
    width: Literal["single", "double"] = "double",
) -> matplotlib.figure.Figure:
    """Demand trajectory with marginal distribution.

    Left panel: time series with mean, +/-1 std band, and optional shock
    marker. Right panel: rotated histogram.
    """
    _apply_style()
    w = _col_width(width)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(w, w / GOLDEN / 1.6),
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05},
    )

    weeks = np.arange(len(demand))
    mu, sigma = demand.mean(), demand.std()

    ax1.fill_between(
        weeks, mu - sigma, mu + sigma,
        color=COLORS["E2"], alpha=0.12, label=r"$\mu \pm \sigma$",
    )
    ax1.plot(weeks, demand, color=COLORS["demand"], linewidth=0.6, label="Demand")
    ax1.axhline(mu, color=COLORS["grid"], linestyle="--", linewidth=0.5)

    if shock_period is not None and shock_period < len(demand):
        ax1.axvline(
            shock_period, color=COLORS["backorder"], linestyle=":",
            linewidth=0.7, label=f"Shock ($t={shock_period}$)",
        )

    ax1.set_xlabel("Week")
    ax1.set_ylabel("Demand (units)")
    ax1.legend(loc="upper left")
    ax1.set_xlim(0, len(demand) - 1)

    ax2.hist(
        demand, bins=25, orientation="horizontal",
        color=COLORS["E2"], alpha=0.6, edgecolor="white", linewidth=0.3,
    )
    ax2.axhline(mu, color=COLORS["grid"], linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Freq.")
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticklabels([])

    return fig


# ── 2. Order quantities across echelons ──────────────────────────────


def plot_order_quantities(
    demand: TimeSeries,
    sim_result: SimulationResult,
    width: Literal["single", "double"] = "double",
) -> matplotlib.figure.Figure:
    """Stacked subplots: customer demand + order quantity per echelon."""
    _apply_style()
    K = len(sim_result.echelon_results)
    w = _col_width(width)
    fig, axes = plt.subplots(
        K + 1, 1, figsize=(w, w * 0.85), sharex=True,
    )
    weeks = np.arange(len(demand))

    # Customer demand
    axes[0].plot(weeks, demand, color=COLORS["demand"], linewidth=0.6)
    axes[0].set_ylabel("Demand")
    axes[0].text(
        0.98, 0.92, "Customer Demand", transform=axes[0].transAxes,
        ha="right", va="top", fontsize=7, fontstyle="italic",
    )

    for k, er in enumerate(sim_result.echelon_results):
        ax = axes[k + 1]
        ax.plot(weeks, er.orders, color=_echelon_color(k), linewidth=0.6)
        ax.axhline(er.orders.mean(), color=COLORS["grid"], linestyle="--", linewidth=0.4)
        ax.set_ylabel("Orders")
        ax.text(
            0.98, 0.92, f"E{k + 1}: {er.name}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, fontstyle="italic",
        )

    axes[-1].set_xlabel("Week")
    axes[0].set_xlim(0, len(demand) - 1)

    return fig


# ── 3. Inventory on-hand across echelons ─────────────────────────────


def plot_inventory_levels(
    sim_result: SimulationResult,
    width: Literal["single", "double"] = "double",
) -> matplotlib.figure.Figure:
    """Inventory on-hand per echelon with zero-line and backorder shading."""
    _apply_style()
    K = len(sim_result.echelon_results)
    w = _col_width(width)
    fig, axes = plt.subplots(K, 1, figsize=(w, w * 0.7), sharex=True)
    if K == 1:
        axes = [axes]

    for k, er in enumerate(sim_result.echelon_results):
        ax = axes[k]
        T = len(er.inventory_levels)
        weeks = np.arange(T)
        inv = er.inventory_levels

        # Shade backorder region
        ax.fill_between(
            weeks, inv, 0,
            where=inv < 0, color=COLORS["backorder"], alpha=0.15,
            label="Backorder",
        )
        ax.fill_between(
            weeks, inv, 0,
            where=inv >= 0, color=COLORS["holding"], alpha=0.10,
            label="On-hand",
        )
        ax.plot(weeks, inv, color=_echelon_color(k), linewidth=0.6)
        ax.axhline(0, color=COLORS["demand"], linewidth=0.4)

        ax.set_ylabel("Inventory")
        ax.text(
            0.98, 0.92,
            f"E{k + 1}: {er.name}  |  FR={er.fill_rate:.0%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
        )
        if k == 0:
            ax.legend(loc="upper left", ncol=2)

    axes[-1].set_xlabel("Week")
    axes[0].set_xlim(0, T - 1)

    return fig


# ── 4. Inventory position (on-hand + pipeline) ──────────────────────


def plot_inventory_position(
    demand: TimeSeries,
    sim_result: SimulationResult,
    chain,
    width: Literal["single", "double"] = "double",
) -> matplotlib.figure.Figure:
    """Inventory position = on-hand + pipeline for each echelon.

    Parameters
    ----------
    chain : SerialSupplyChain
        The chain object *after* simulate() has been called, so that
        echelon pipeline state is available.
    """
    _apply_style()
    K = len(sim_result.echelon_results)
    w = _col_width(width)
    fig, axes = plt.subplots(K, 1, figsize=(w, w * 0.7), sharex=True)
    if K == 1:
        axes = [axes]

    for k, er in enumerate(sim_result.echelon_results):
        ax = axes[k]
        T = len(er.inventory_levels)
        weeks = np.arange(T)

        # Reconstruct inventory position from on-hand + pipeline snapshot
        # We can approximate IP from orders and inventory arrays:
        # IP(t) = inv(t) + sum of orders in pipeline at time t
        inv = er.inventory_levels
        orders = er.orders
        L = chain.echelons[k].lead_time

        # Build pipeline sum: at each t, pipeline = orders[t-L+1 .. t]
        pipeline_sum = np.zeros(T)
        for t in range(T):
            start = max(0, t - L + 1)
            pipeline_sum[t] = orders[start : t + 1].sum()

        ip = inv + pipeline_sum

        ax.plot(weeks, ip, color=_echelon_color(k), linewidth=0.6, label="Inv. Position")
        ax.plot(weeks, inv, color=_echelon_color(k), linewidth=0.4, linestyle="--",
                alpha=0.5, label="On-hand only")
        ax.axhline(0, color=COLORS["demand"], linewidth=0.4)

        ax.set_ylabel("Units")
        ax.text(
            0.98, 0.92, f"E{k + 1}: {er.name} (L={L})",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
        )
        if k == 0:
            ax.legend(loc="upper left", ncol=2)

    axes[-1].set_xlabel("Week")
    axes[0].set_xlim(0, T - 1)

    return fig


# ── 5. Order streams overlay ────────────────────────────────────────


def plot_order_streams(
    demand: TimeSeries,
    sim_result: SimulationResult,
    echelon_indices: list[int] | None = None,
    width: Literal["single", "double"] = "double",
) -> matplotlib.figure.Figure:
    """All echelon order streams overlaid on customer demand."""
    _apply_style()
    if echelon_indices is None:
        echelon_indices = list(range(len(sim_result.echelon_results)))

    w = _col_width(width)
    fig, ax = plt.subplots(figsize=(w, w / GOLDEN / 1.4))
    weeks = np.arange(len(demand))

    ax.plot(
        weeks, demand, color=COLORS["demand"], linewidth=1.0,
        label="Customer Demand", zorder=10,
    )
    for idx in echelon_indices:
        er = sim_result.echelon_results[idx]
        ax.plot(
            weeks, er.orders, color=_echelon_color(idx), linewidth=0.5,
            alpha=0.8, label=f"E{idx + 1}: {er.name}",
        )

    ax.set_xlabel("Week")
    ax.set_ylabel("Quantity (units)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_xlim(0, len(demand) - 1)

    return fig


# ── 6. Cost time series per echelon ─────────────────────────────────


def plot_cost_timeseries(
    sim_result: SimulationResult,
    width: Literal["single", "double"] = "double",
) -> matplotlib.figure.Figure:
    """Per-period cost for each echelon, stacked vertically."""
    _apply_style()
    K = len(sim_result.echelon_results)
    w = _col_width(width)
    fig, axes = plt.subplots(K, 1, figsize=(w, w * 0.7), sharex=True)
    if K == 1:
        axes = [axes]

    for k, er in enumerate(sim_result.echelon_results):
        ax = axes[k]
        T = len(er.costs)
        weeks = np.arange(T)
        inv = er.inventory_levels

        holding_mask = inv >= 0
        costs_h = np.where(holding_mask, er.costs, 0)
        costs_b = np.where(~holding_mask, er.costs, 0)

        ax.bar(weeks, costs_h, width=1.0, color=COLORS["holding"], alpha=0.6, label="Holding")
        ax.bar(weeks, costs_b, width=1.0, color=COLORS["backorder"], alpha=0.6, label="Backorder")

        ax.set_ylabel("Cost")
        ax.text(
            0.98, 0.92,
            f"E{k + 1}: {er.name}  |  Total={er.total_cost:,.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
        )
        if k == 0:
            ax.legend(loc="upper left", ncol=2)

    axes[-1].set_xlabel("Week")
    axes[0].set_xlim(0, T - 1)

    return fig


# ── 7. Cost decomposition bar chart ─────────────────────────────────


def plot_cost_decomposition(
    results_by_model: dict[str, SimulationResult],
    width: Literal["single", "double"] = "double",
) -> matplotlib.figure.Figure:
    """Stacked bar: holding vs backorder cost per model."""
    _apply_style()
    w = _col_width(width)
    fig, ax = plt.subplots(figsize=(w, w / GOLDEN / 1.4))

    names = list(results_by_model.keys())
    x = np.arange(len(names))

    holding_costs = []
    backorder_costs = []
    for result in results_by_model.values():
        h_total = 0.0
        b_total = 0.0
        for er in result.echelon_results:
            inv = er.inventory_levels
            costs = er.costs
            h_mask = inv >= 0
            h_total += float(costs[h_mask].sum())
            b_total += float(costs[~h_mask].sum())
        holding_costs.append(h_total)
        backorder_costs.append(b_total)

    bar_w = 0.6
    ax.bar(x, holding_costs, bar_w, label="Holding", color=COLORS["holding"], edgecolor="white", linewidth=0.3)
    ax.bar(x, backorder_costs, bar_w, bottom=holding_costs, label="Backorder",
           color=COLORS["backorder"], edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Total Cost")
    ax.legend(loc="upper right")

    return fig


# ── 8. Bullwhip amplification ───────────────────────────────────────


def plot_bullwhip_amplification(
    results_by_model: dict[str, SimulationResult],
    echelon_labels: list[str] | None = None,
    width: Literal["single", "double"] = "single",
) -> matplotlib.figure.Figure:
    """Log-scale cumulative bullwhip ratio across echelons."""
    _apply_style()
    w = _col_width(width)
    fig, ax = plt.subplots(figsize=(w, w / GOLDEN))

    for name, result in results_by_model.items():
        bw_ratios = [er.bullwhip_ratio for er in result.echelon_results]
        cum_bw = np.cumprod(bw_ratios)
        x = np.arange(1, len(cum_bw) + 1)
        ax.plot(x, cum_bw, marker="o", markersize=4, linewidth=1.0, label=name)

    ax.set_yscale("log")
    ax.set_xlabel("Echelon")
    ax.set_ylabel("Cumulative Bullwhip Ratio")
    if echelon_labels:
        ax.set_xticks(range(1, len(echelon_labels) + 1))
        ax.set_xticklabels(echelon_labels)
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.2, linewidth=0.3)

    return fig


# ── 9. Summary dashboard ────────────────────────────────────────────


def plot_summary_dashboard(
    demand: TimeSeries,
    sim_result: SimulationResult,
    width: Literal["single", "double"] = "double",
) -> matplotlib.figure.Figure:
    """4-panel dashboard: demand, orders overlay, inventory, BW ratios."""
    _apply_style()
    w = _col_width(width)
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.75))
    weeks = np.arange(len(demand))
    K = len(sim_result.echelon_results)

    # (a) Demand
    ax = axes[0, 0]
    ax.plot(weeks, demand, color=COLORS["demand"], linewidth=0.6)
    ax.axhline(demand.mean(), color=COLORS["grid"], linestyle="--", linewidth=0.4)
    ax.set_ylabel("Demand")
    ax.set_title("(a) Customer Demand", fontsize=8)

    # (b) Order streams
    ax = axes[0, 1]
    ax.plot(weeks, demand, color=COLORS["demand"], linewidth=0.8, label="Demand")
    for k, er in enumerate(sim_result.echelon_results):
        ax.plot(weeks, er.orders, color=_echelon_color(k), linewidth=0.4,
                alpha=0.7, label=f"E{k + 1}")
    ax.set_ylabel("Orders")
    ax.set_title("(b) Order Streams", fontsize=8)
    ax.legend(loc="upper left", ncol=3, fontsize=6)

    # (c) Inventory
    ax = axes[1, 0]
    for k, er in enumerate(sim_result.echelon_results):
        ax.plot(weeks, er.inventory_levels, color=_echelon_color(k),
                linewidth=0.5, label=f"E{k + 1}")
    ax.axhline(0, color=COLORS["demand"], linewidth=0.4)
    ax.set_xlabel("Week")
    ax.set_ylabel("Inventory")
    ax.set_title("(c) Inventory On-Hand", fontsize=8)
    ax.legend(loc="lower left", ncol=2, fontsize=6)

    # (d) Bullwhip & fill rate
    ax = axes[1, 1]
    bw = [er.bullwhip_ratio for er in sim_result.echelon_results]
    fr = [er.fill_rate for er in sim_result.echelon_results]
    x = np.arange(1, K + 1)
    labels = [f"E{k + 1}" for k in range(K)]

    bar_w = 0.35
    bars1 = ax.bar(x - bar_w / 2, bw, bar_w, color=COLORS["E1"], label="BW Ratio")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + bar_w / 2, fr, bar_w, color=COLORS["E3"], alpha=0.7, label="Fill Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bullwhip Ratio")
    ax2.set_ylabel("Fill Rate")
    ax2.set_ylim(0, 1.1)
    ax.set_title("(d) Bullwhip & Fill Rate", fontsize=8)

    # Combined legend
    lines = [bars1, bars2]
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, loc="upper left", fontsize=6)

    for a in [axes[0, 0], axes[0, 1]]:
        a.set_xlim(0, len(demand) - 1)
    axes[1, 0].set_xlim(0, len(demand) - 1)

    return fig


# ── 10. Per-echelon detail panel ─────────────────────────────────────


def plot_echelon_detail(
    demand: TimeSeries,
    sim_result: SimulationResult,
    echelon_index: int = 0,
    width: Literal["single", "double"] = "double",
) -> matplotlib.figure.Figure:
    """3-row detail for one echelon: orders vs demand, inventory, costs."""
    _apply_style()
    er = sim_result.echelon_results[echelon_index]
    w = _col_width(width)
    fig, axes = plt.subplots(3, 1, figsize=(w, w * 0.65), sharex=True)
    weeks = np.arange(len(demand))
    color = _echelon_color(echelon_index)

    # Orders vs demand
    ax = axes[0]
    ax.plot(weeks, demand, color=COLORS["demand"], linewidth=0.6, label="Demand")
    ax.plot(weeks, er.orders, color=color, linewidth=0.6, label=f"E{echelon_index + 1} Orders")
    ax.set_ylabel("Quantity")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title(f"Echelon {echelon_index + 1}: {er.name}", fontsize=9)

    # Inventory
    ax = axes[1]
    ax.fill_between(weeks, er.inventory_levels, 0,
                    where=er.inventory_levels >= 0, color=COLORS["holding"], alpha=0.15)
    ax.fill_between(weeks, er.inventory_levels, 0,
                    where=er.inventory_levels < 0, color=COLORS["backorder"], alpha=0.15)
    ax.plot(weeks, er.inventory_levels, color=color, linewidth=0.6)
    ax.axhline(0, color=COLORS["demand"], linewidth=0.4)
    ax.set_ylabel("Inventory")

    # Costs
    ax = axes[2]
    inv = er.inventory_levels
    holding_mask = inv >= 0
    ax.bar(weeks[holding_mask], er.costs[holding_mask], width=1.0,
           color=COLORS["holding"], alpha=0.6, label="Holding")
    ax.bar(weeks[~holding_mask], er.costs[~holding_mask], width=1.0,
           color=COLORS["backorder"], alpha=0.6, label="Backorder")
    ax.set_ylabel("Cost")
    ax.set_xlabel("Week")
    ax.legend(loc="upper left", ncol=2)

    axes[0].set_xlim(0, len(demand) - 1)

    return fig
