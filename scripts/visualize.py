#!/usr/bin/env python3
"""Generate all publication-grade diagnostic figures.

Usage:
    python scripts/visualize.py              # show interactively
    python scripts/visualize.py --save       # save PDFs to figures/
    python scripts/visualize.py --save --dpi 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deepbullwhip import SemiconductorDemandGenerator, SerialSupplyChain
from deepbullwhip.diagnostics.plots import (
    plot_bullwhip_amplification,
    plot_cost_decomposition,
    plot_cost_timeseries,
    plot_demand_trajectory,
    plot_echelon_detail,
    plot_inventory_levels,
    plot_inventory_position,
    plot_order_quantities,
    plot_order_streams,
    plot_summary_dashboard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagnostic figures")
    parser.add_argument("--save", action="store_true", help="Save figures as PDF")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, default=156)
    args = parser.parse_args()

    # ── Generate demand & simulate ───────────────────────────────────
    gen = SemiconductorDemandGenerator()
    demand = gen.generate(T=args.T, seed=args.seed)
    chain = SerialSupplyChain()
    forecasts_mean = np.full_like(demand, demand.mean())
    forecasts_std = np.full_like(demand, demand.std())
    result = chain.simulate(demand, forecasts_mean, forecasts_std)

    # ── Generate all figures ─────────────────────────────────────────
    figures = {
        "fig1_demand_trajectory": plot_demand_trajectory(demand, shock_period=104),
        "fig2_order_quantities": plot_order_quantities(demand, result),
        "fig3_order_streams": plot_order_streams(demand, result),
        "fig4_inventory_levels": plot_inventory_levels(result),
        "fig5_inventory_position": plot_inventory_position(demand, result, chain),
        "fig6_cost_timeseries": plot_cost_timeseries(result),
        "fig7_cost_decomposition": plot_cost_decomposition({"Default (OUT)": result}),
        "fig8_bullwhip_amplification": plot_bullwhip_amplification(
            {"Default (OUT)": result},
            echelon_labels=["Distributor", "OSAT", "Foundry", "Supplier"],
        ),
        "fig9_summary_dashboard": plot_summary_dashboard(demand, result),
    }

    # Per-echelon detail
    for k in range(len(result.echelon_results)):
        name = result.echelon_results[k].name.lower()
        figures[f"fig10_{name}_detail"] = plot_echelon_detail(demand, result, k)

    # ── Print summary metrics ────────────────────────────────────────
    print("=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    for k, er in enumerate(result.echelon_results):
        print(
            f"  E{k+1} {er.name:12s}  |  BW={er.bullwhip_ratio:6.2f}  "
            f"FR={er.fill_rate:.1%}  Cost={er.total_cost:10,.1f}"
        )
    print(f"  {'':12s}       |  BW_cum={result.cumulative_bullwhip:6.2f}  "
          f"Total Cost={result.total_cost:10,.1f}")
    print("=" * 60)

    # ── Save or show ─────────────────────────────────────────────────
    if args.save:
        outdir = Path(args.outdir)
        outdir.mkdir(exist_ok=True)
        for name, fig in figures.items():
            path = outdir / f"{name}.pdf"
            fig.savefig(path, dpi=args.dpi)
            print(f"  Saved {path}")
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
