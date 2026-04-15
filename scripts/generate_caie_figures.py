"""
Generate all figures for the DeepBullwhip CAIE 2026 paper.

Paper:  Arief (2026), "Accuracy-Robustness Tradeoffs in ML-Driven
        Semiconductor Supply Chains", Computers & Industrial Engineering.

Requires: deepbullwhip>=0.4.0, numpy, matplotlib
Usage:    python scripts/generate_caie_figures.py [--outdir figures/arief2026/]

Produces 14 figures (PNG 300dpi + PDF):
  Main body (9):  summary dashboard, MC BWR distribution, lead-time
      sensitivity, scalability, policy comparison, synthetic vs real,
      forecaster comparison, cross-chain, cost ratio sensitivity.
  Appendix (5):   order quantities, inventory levels, cost timeseries,
      echelon detail (foundry), echelon detail (wafer).
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepbullwhip import (
    SemiconductorDemandGenerator,
    SerialSupplyChain,
    VectorizedSupplyChain,
    EchelonConfig,
)
from deepbullwhip.diagnostics.plots import (
    plot_summary_dashboard,
    plot_echelon_detail,
    plot_order_quantities,
    plot_inventory_levels,
    plot_cost_timeseries,
)
from deepbullwhip.benchmark import BenchmarkRunner

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})
C = ["#006747", "#c4a35a", "#004040", "#9e8340", "#8B0000"]


def save(fig, outdir, name):
    """Save figure in both PNG and PDF, then close."""
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


def main(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # ── Shared data ───────────────────────────────────────────────────
    gen = SemiconductorDemandGenerator()
    demand = gen.generate(T=156, seed=42)
    chain = SerialSupplyChain()
    fm = np.full_like(demand, demand.mean())
    fs = np.full_like(demand, demand.std())
    result = chain.simulate(demand, fm, fs)

    print("Main-body figures:")

    # ── Fig 1: Summary dashboard ──────────────────────────────────────
    fig = plot_summary_dashboard(demand, result)
    if fig._suptitle:
        fig._suptitle.set_visible(False)
    save(fig, outdir, "fig_summary_dashboard")

    # ── Fig 2: MC BWR distribution ────────────────────────────────────
    demand_batch = gen.generate_batch(T=156, n_paths=1000, seed=42)
    vc = VectorizedSupplyChain()
    fm_b = np.full_like(demand_batch, demand_batch.mean())
    fs_b = np.full_like(demand_batch, demand_batch.std())
    mc = vc.simulate(demand_batch, fm_b, fs_b)
    bwr_arr = np.array([
        [er.bullwhip_ratio for er in mc.to_simulation_result(p).echelon_results]
        for p in range(1000)
    ])

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
    names = ["E1: Distributor", "E2: Assembly", "E3: Foundry", "E4: Wafer"]
    for i, ax in enumerate(axes):
        ax.hist(bwr_arr[:, i], bins=30, color=C[i], alpha=0.85,
                edgecolor="white", linewidth=0.5)
        med = np.median(bwr_arr[:, i])
        ax.axvline(med, color="#c0392b", ls="--", lw=1, label=f"Med={med:.1f}")
        ax.set_title(names[i], fontsize=8, fontweight="bold")
        ax.set_xlabel("BWR", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Count", fontsize=7)
    fig.tight_layout()
    save(fig, outdir, "fig_mc_bwr_distribution")

    # ── Fig 3: Lead time sensitivity ──────────────────────────────────
    scenarios = {
        "Short L=[1,2,4,2]": [1, 2, 4, 2],
        "Baseline L=[2,4,12,8]": [2, 4, 12, 8],
        "Long L=[4,8,20,12]": [4, 8, 20, 12],
    }
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for label, lts in scenarios.items():
        configs = [
            EchelonConfig(f"E{k+1}", lead_time=lts[k],
                          holding_cost=[.15, .12, .08, .05][k],
                          backorder_cost=[.60, .50, .40, .30][k])
            for k in range(4)
        ]
        ch = SerialSupplyChain.from_config(configs)
        r = ch.simulate(demand, fm, fs)
        cum = np.cumprod([er.bullwhip_ratio for er in r.echelon_results])
        ax.semilogy(range(1, 5), cum, "o-", label=label, markersize=5, linewidth=1.5)
    ax.set_xlabel("Echelon", fontsize=8)
    ax.set_ylabel("Cumulative BWR", fontsize=8)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["E1", "E2", "E3", "E4"], fontsize=7)
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.2, which="both")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, outdir, "fig_leadtime_sensitivity")

    # ── Fig 4: Scalability ────────────────────────────────────────────
    N_vals = [10, 50, 100, 500, 1000, 2000, 5000]
    ser_N, vec_N = [], []
    for N in N_vals:
        db = gen.generate_batch(T=156, n_paths=N, seed=42)
        fmb = np.full_like(db, db.mean())
        fsb = np.full_like(db, db.std())
        ns = min(20, N)
        ch = SerialSupplyChain()
        t0 = time.perf_counter()
        for p in range(ns):
            ch.simulate(db[p], fmb[p], fsb[p])
        ser_N.append((time.perf_counter() - t0) * (N / ns))
        v = VectorizedSupplyChain()
        t0 = time.perf_counter()
        v.simulate(db, fmb, fsb)
        vec_N.append(time.perf_counter() - t0)

    T_vals = [52, 156, 520, 1040, 2080]
    ser_T, vec_T = [], []
    for T in T_vals:
        db = gen.generate_batch(T=T, n_paths=500, seed=42)
        fmb = np.full_like(db, db.mean())
        fsb = np.full_like(db, db.std())
        ch = SerialSupplyChain()
        t0 = time.perf_counter()
        for p in range(20):
            ch.simulate(db[p], fmb[p], fsb[p])
        ser_T.append((time.perf_counter() - t0) * 25)
        v = VectorizedSupplyChain()
        t0 = time.perf_counter()
        v.simulate(db, fmb, fsb)
        vec_T.append(time.perf_counter() - t0)

    K_vals = [2, 4, 6, 8, 10, 12]
    ser_K, vec_K = [], []
    for K in K_vals:
        configs = [
            EchelonConfig(f"E{k+1}", lead_time=max(1, 2 + k),
                          holding_cost=max(.02, .15 - .01 * k),
                          backorder_cost=max(.10, .60 - .03 * k))
            for k in range(K)
        ]
        db = gen.generate_batch(T=156, n_paths=500, seed=42)
        fmb = np.full_like(db, db.mean())
        fsb = np.full_like(db, db.std())
        ch = SerialSupplyChain.from_config(configs)
        t0 = time.perf_counter()
        for p in range(20):
            ch.simulate(db[p], fmb[p], fsb[p])
        ser_K.append((time.perf_counter() - t0) * 25)
        v = VectorizedSupplyChain(configs)
        t0 = time.perf_counter()
        v.simulate(db, fmb, fsb)
        vec_K.append(time.perf_counter() - t0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, xv, sy, vy, xlab, title, logx in [
        (axes[0], N_vals, ser_N, vec_N, "$N$ (paths)", "(a) Scaling with $N$", True),
        (axes[1], T_vals, ser_T, vec_T, "$T$ (periods)", "(b) Scaling with $T$", True),
        (axes[2], K_vals, ser_K, vec_K, "$K$ (echelons)", "(c) Scaling with $K$", False),
    ]:
        if logx:
            ax.loglog(xv, sy, "o-", color=C[1], lw=1.5, ms=4, label="Serial")
            ax.loglog(xv, vy, "s-", color=C[0], lw=1.5, ms=4, label="Vectorized")
        else:
            ax.semilogy(xv, sy, "o-", color=C[1], lw=1.5, ms=4, label="Serial")
            ax.semilogy(xv, vy, "s-", color=C[0], lw=1.5, ms=4, label="Vectorized")
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel("Wall time (s)", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, outdir, "fig_scalability")

    # ── Fig 5: Policy comparison ──────────────────────────────────────
    runner = BenchmarkRunner(
        "semiconductor_4tier", "semiconductor_ar1", T=156, N=500, seed=42
    )
    r5a = runner.run(
        policies=["order_up_to", ("proportional_out", {"alpha": 0.3}),
                  "smoothing_out", "constant_order"],
        metrics=["CUM_BWR", "TC"],
    )
    e4 = r5a[r5a["echelon"] == "E4"]
    pols = ["order_up_to", "proportional_out", "smoothing_out", "constant_order"]
    labs = ["OUT", "POUT(0.3)", "Smoothing", "Constant"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    bwr_v = [
        e4[(e4["policy"] == p) & (e4["metric"] == "CUM_BWR")]["value"].values[0]
        for p in pols
    ]
    tc_v = [
        e4[(e4["policy"] == p) & (e4["metric"] == "TC")]["value"].values[0]
        for p in pols
    ]
    ax1.bar(range(4), bwr_v, color=C[:4], alpha=0.85, edgecolor="white")
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labs, fontsize=7, rotation=10)
    ax1.set_ylabel("Cumulative BWR", fontsize=8)
    ax1.set_yscale("log")
    ax1.tick_params(labelsize=6)
    ax1.set_title("(a) Bullwhip amplification", fontsize=9, fontweight="bold")

    ax2.bar(range(4), tc_v, color=C[:4], alpha=0.85, edgecolor="white")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labs, fontsize=7, rotation=10)
    ax2.set_ylabel("Total Cost (E4)", fontsize=8)
    ax2.tick_params(labelsize=6)
    ax2.set_title("(b) Cost at Wafer Supplier", fontsize=9, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "fig_policy_comparison")

    # ── Fig 6: Synthetic vs real ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.8, 3))
    ax.bar([0, 1], [429, 66076], color=[C[0], C[2]], alpha=0.85, edgecolor="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Synthetic AR(1)", "WSTS Real Data"], fontsize=7)
    ax.set_ylabel("Cumulative BWR at E4", fontsize=8)
    ax.set_yscale("log")
    ax.tick_params(labelsize=6)
    for i, v in enumerate([429, 66076]):
        ax.text(i, v * 1.5, f"{v:,.0f}", ha="center", fontsize=7, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "fig_synthetic_vs_real")

    # ── Fig 7: Forecaster comparison ──────────────────────────────────
# ── Fig 7: Forecaster comparison ──────────────────────────────────
    forecaster_specs = ["naive", ("moving_average", {"window": 10}),
                        ("exponential_smoothing", {"alpha": 0.3})]
    fc_labs = ["Naive", "MA(10)", "SES(0.3)"]

    # Optionally include DeepAR if GluonTS is available
    try:
        from deepbullwhip.forecast.deepar import DeepARTrainer
        print("  Training DeepAR for forecaster comparison figure...")
        train_series = [gen.generate(T=260, seed=s) for s in range(200)]
        trainer = DeepARTrainer(
            freq="W", prediction_length=1, context_length=52,
            epochs=30, num_layers=3, hidden_size=40,
        )
        deepar_fc = trainer.train(train_series)
        from deepbullwhip.registry import _REGISTRY
        _REGISTRY["forecaster"]["deepar"] = lambda **kw: deepar_fc
        forecaster_specs.append("deepar")
        fc_labs.append("DeepAR")
        print("  DeepAR ready.")
    except ImportError:
        print("  [SKIP] GluonTS not installed — figure without DeepAR.")

    r5b = runner.run(
        policies=["order_up_to"],
        forecasters=forecaster_specs,
        metrics=["CUM_BWR", "FILL_RATE"],
    )
    e4b = r5b[r5b["echelon"] == "E4"]
    fc_list = list(dict.fromkeys(r5b["forecaster"].values))
    n_fc = len(fc_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    bwr_fc = [
        e4b[(e4b["forecaster"] == f) & (e4b["metric"] == "CUM_BWR")]["value"].values[0]
        for f in fc_list
    ]
    fr_fc = [
        e4b[(e4b["forecaster"] == f) & (e4b["metric"] == "FILL_RATE")]["value"].values[0]
        for f in fc_list
    ]
    colors = C[:n_fc] if n_fc <= len(C) else C + ["#555555"] * (n_fc - len(C))
    ax1.bar(range(n_fc), bwr_fc, color=colors, alpha=0.85, edgecolor="white")
    ax1.set_xticks(range(n_fc))
    ax1.set_xticklabels(fc_labs[:n_fc], fontsize=7)
    ax1.set_ylabel("Cumulative BWR (E4)", fontsize=8)
    ax1.tick_params(labelsize=6)
    ax1.set_title("(a) BWR by forecaster", fontsize=9, fontweight="bold")

    ax2.bar(range(n_fc), fr_fc, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(range(n_fc))
    ax2.set_xticklabels(fc_labs[:n_fc], fontsize=7)
    ax2.set_ylabel("Fill Rate (E4)", fontsize=8)
    ax2.tick_params(labelsize=6)
    ax2.set_title("(b) Fill rate by forecaster", fontsize=9, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "fig_forecaster_comparison")

    # ── Fig 8: Cross-chain ────────────────────────────────────────────
    chain_data = {
        "Semiconductor\n4-tier": 429,
        "Beer Game\n4-tier": 10.0,
        "Consumer\n2-tier": 2.5,
    }
    fig, ax = plt.subplots(figsize=(3.8, 3))
    ax.bar(range(3), list(chain_data.values()), color=C[:3],
           alpha=0.85, edgecolor="white")
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(chain_data.keys()), fontsize=7)
    ax.set_ylabel("Cumulative BWR", fontsize=8)
    ax.set_yscale("log")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, outdir, "fig_cross_chain")

    # ── Fig 9: Cost ratio sensitivity ─────────────────────────────────
    ratios = [2, 4, 6, 8, 10]
    costs = []
    for r in ratios:
        h = 1 / (1 + r)
        b = r / (1 + r)
        configs = [
            EchelonConfig(f"E{k+1}", lead_time=[2, 4, 12, 8][k],
                          holding_cost=h, backorder_cost=b)
            for k in range(4)
        ]
        ch = SerialSupplyChain.from_config(configs)
        res = ch.simulate(demand, fm, fs)
        costs.append(res.total_cost)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(range(len(ratios)), costs, color=C[1], alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([str(r) for r in ratios], fontsize=7)
    ax.set_xlabel("Cost Ratio $b/h$", fontsize=8)
    ax.set_ylabel("Total Supply Chain Cost", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, outdir, "fig_cost_ratio_sensitivity")

    print("\nAppendix diagnostic figures:")

    # ── Appendix: Order quantities ────────────────────────────────────
    fig = plot_order_quantities(demand, result, width="double")
    if fig._suptitle:
        fig._suptitle.set_visible(False)
    save(fig, outdir, "fig_order_quantities")

    # ── Appendix: Inventory levels ────────────────────────────────────
    fig = plot_inventory_levels(result, width="double")
    if fig._suptitle:
        fig._suptitle.set_visible(False)
    save(fig, outdir, "fig_inventory_levels")

    # ── Appendix: Cost timeseries ─────────────────────────────────────
    fig = plot_cost_timeseries(result, width="double")
    if fig._suptitle:
        fig._suptitle.set_visible(False)
    save(fig, outdir, "fig_cost_timeseries")

    # ── Appendix: Echelon details ─────────────────────────────────────
    for idx, name in [(2, "foundry"), (3, "wafer")]:
        fig = plot_echelon_detail(demand, result, echelon_index=idx, width="double")
        if fig._suptitle:
            fig._suptitle.set_visible(False)
        save(fig, outdir, f"fig_echelon_detail_{name}")

    elapsed = time.perf_counter() - t_start
    print(f"\nAll 14 figures generated in {elapsed:.1f}s")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CAIE paper figures")
    parser.add_argument("--outdir", type=str, default="figures/",
                        help="Output directory for figures")
    args = parser.parse_args()
    main(Path(args.outdir))
