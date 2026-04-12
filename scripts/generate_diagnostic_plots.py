"""Generate all publication-grade figures for the DeepBullwhip white paper."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deepbullwhip import (
    SemiconductorDemandGenerator,
    SerialSupplyChain,
    VectorizedSupplyChain,
    EchelonConfig,
)
from deepbullwhip.diagnostics.plots import (
    plot_demand_trajectory,
    plot_order_quantities,
    plot_inventory_levels,
    plot_cost_timeseries,
    plot_cost_decomposition,
    plot_bullwhip_amplification,
    plot_summary_dashboard,
    plot_echelon_detail,
)

# ── 1. Generate demand and simulate ────────────────────────────────────
gen = SemiconductorDemandGenerator()
demand = gen.generate(T=156, seed=42)

chain = SerialSupplyChain()
fm = np.full_like(demand, demand.mean())
fs = np.full_like(demand, demand.std())
result = chain.simulate(demand, fm, fs)

print("=== Single-Path Simulation Metrics ===")
for k, er in enumerate(result.echelon_results):
    print(f"  E{k+1}: {er.name:18s}  BWR={er.bullwhip_ratio:.3f}  "
          f"FR={er.fill_rate:.1%}  TotalCost={er.total_cost:,.0f}")

# ── 2. Built-in plots ──────────────────────────────────────────────────
fig1 = plot_demand_trajectory(demand, width="double")
fig1.savefig("figures/fig_demand_trajectory.pdf", dpi=300)
fig1.savefig("figures/fig_demand_trajectory.png", dpi=300)
print("Saved: fig_demand_trajectory")

fig2 = plot_order_quantities(demand, result, width="double")
fig2.savefig("figures/fig_order_quantities.pdf", dpi=300)
fig2.savefig("figures/fig_order_quantities.png", dpi=300)
print("Saved: fig_order_quantities")

fig3 = plot_inventory_levels(result, width="double")
fig3.savefig("figures/fig_inventory_levels.pdf", dpi=300)
fig3.savefig("figures/fig_inventory_levels.png", dpi=300)
print("Saved: fig_inventory_levels")

fig4 = plot_cost_timeseries(result, width="double")
fig4.savefig("figures/fig_cost_timeseries.pdf", dpi=300)
fig4.savefig("figures/fig_cost_timeseries.png", dpi=300)
print("Saved: fig_cost_timeseries")

fig5 = plot_cost_decomposition({"Default": result}, width="single")
fig5.savefig("figures/fig_cost_decomposition.pdf", dpi=300)
fig5.savefig("figures/fig_cost_decomposition.png", dpi=300)
print("Saved: fig_cost_decomposition")

fig6 = plot_bullwhip_amplification({"Default": result}, width="single")
fig6.savefig("figures/fig_bullwhip_amplification.pdf", dpi=300)
fig6.savefig("figures/fig_bullwhip_amplification.png", dpi=300)
print("Saved: fig_bullwhip_amplification")

fig7 = plot_summary_dashboard(demand, result)
fig7.savefig("figures/fig_summary_dashboard.pdf", dpi=300)
fig7.savefig("figures/fig_summary_dashboard.png", dpi=300)
print("Saved: fig_summary_dashboard")

fig8 = plot_echelon_detail(demand, result, echelon_index=2, width="double")
fig8.savefig("figures/fig_echelon_detail_foundry.pdf", dpi=300)
fig8.savefig("figures/fig_echelon_detail_foundry.png", dpi=300)
print("Saved: fig_echelon_detail_foundry")
plt.close('all')

# ── 3. Supply chain schematic ──────────────────────────────────────────
fig_sc, ax = plt.subplots(figsize=(7.0, 2.5))
ax.set_xlim(-0.5, 11.5); ax.set_ylim(0, 3)
ax.axis('off')

echelons = [
    ("E4\nWafer/Material", 1.0),
    ("E3\nFoundry/Fab", 3.5),
    ("E2\nAssembly/Test", 6.0),
    ("E1\nDistributor/OEM", 8.5),
]
for name, x in echelons:
    rect = plt.Rectangle((x-0.7, 0.8), 1.4, 1.4, facecolor='#e8f5f0',
                           edgecolor='#006747', linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x, 1.5, name, ha='center', va='center', fontsize=7,
            fontfamily='serif', fontweight='bold', color='#006747', zorder=3)

for i in range(len(echelons)-1):
    x1 = echelons[i][1] + 0.7
    x2 = echelons[i+1][1] - 0.7
    mid = (x1+x2)/2
    ax.annotate('', xy=(x1, 1.9), xytext=(x2, 1.9),
                arrowprops=dict(arrowstyle='->', color='#c4a35a', lw=1.5))
    ax.text(mid, 2.15, 'Orders $O_k(t)$', ha='center', va='bottom', fontsize=5.5,
            color='#c4a35a', fontfamily='serif', style='italic')
    ax.annotate('', xy=(x2, 1.1), xytext=(x1, 1.1),
                arrowprops=dict(arrowstyle='->', color='#004040', lw=1.5))
    ax.text(mid, 0.85, 'Goods $R_k(t)$', ha='center', va='top', fontsize=5.5,
            color='#004040', fontfamily='serif', style='italic')

ax.annotate('', xy=(9.2+0.3, 1.5), xytext=(9.8, 1.5),
            arrowprops=dict(arrowstyle='->', color='#1a1a1a', lw=1.5))
ax.text(10.1, 1.5, 'Customer\nDemand $D(t)$', ha='left', va='center', fontsize=6,
        fontfamily='serif', color='#1a1a1a')

for name, x in echelons:
    ax.text(x, 0.6, '$I_k, S_k, C_k$', ha='center', va='top', fontsize=5.5,
            fontfamily='serif', color='#666666', style='italic')

ax.set_title('Serial Supply Chain: Information and Material Flows',
             fontsize=9, fontfamily='serif', fontweight='bold', pad=10)
fig_sc.tight_layout()
fig_sc.savefig("figures/fig_supply_chain_schematic.pdf", dpi=300)
fig_sc.savefig("figures/fig_supply_chain_schematic.png", dpi=300)
print("Saved: fig_supply_chain_schematic")
plt.close('all')

# ── 4. Monte Carlo BWR distributions ─────────────────────────────────
print("\n=== Monte Carlo Experiment (1000 paths) ===")
demand_batch = gen.generate_batch(T=156, n_paths=1000, seed=42)
vchain = VectorizedSupplyChain()
fm_batch = np.full_like(demand_batch, demand_batch.mean())
fs_batch = np.full_like(demand_batch, demand_batch.std())
mc_result = vchain.simulate(demand_batch, fm_batch, fs_batch)
print(f"  Mean metrics: {mc_result.mean_metrics()}")

bwr_per_path = []
for p in range(1000):
    sr = mc_result.to_simulation_result(path_index=p)
    bwr_per_path.append([er.bullwhip_ratio for er in sr.echelon_results])
bwr_arr = np.array(bwr_per_path)

fig_mc, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True)
echelon_names = ["E1: Distributor", "E2: Assembly", "E3: Foundry", "E4: Wafer"]
colors = ['#006747', '#c4a35a', '#004040', '#9e8340']
for i, ax in enumerate(axes):
    ax.hist(bwr_arr[:,i], bins=30, color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.3)
    med = np.median(bwr_arr[:,i])
    ax.axvline(med, color='red', linestyle='--', linewidth=0.8, label=f'Med={med:.2f}')
    ax.set_title(echelon_names[i], fontsize=7, fontfamily='serif')
    ax.set_xlabel('BWR', fontsize=6, fontfamily='serif')
    ax.legend(fontsize=5)
    ax.tick_params(labelsize=5)
    print(f"  {echelon_names[i]}: mean={np.mean(bwr_arr[:,i]):.2f}, med={med:.2f}, std={np.std(bwr_arr[:,i]):.2f}")
axes[0].set_ylabel('Count', fontsize=6, fontfamily='serif')
fig_mc.suptitle('Bullwhip Ratio Distribution (N=1,000 Monte Carlo Paths)', fontsize=8, fontfamily='serif', fontweight='bold')
fig_mc.tight_layout()
fig_mc.savefig("figures/fig_mc_bwr_distribution.pdf", dpi=300)
fig_mc.savefig("figures/fig_mc_bwr_distribution.png", dpi=300)
print("Saved: fig_mc_bwr_distribution")
plt.close('all')

# ── 5. Lead time sensitivity ──────────────────────────────────────────
print("\n=== Lead Time Sensitivity ===")
lt_scenarios = {
    "Short": [1, 2, 4, 2],
    "Baseline": [2, 4, 12, 8],
    "Long":  [4, 8, 20, 12],
}

fig_lt, ax = plt.subplots(figsize=(3.5, 2.5))
markers = ['s', 'o', '^']
lt_colors = ['#c4a35a', '#006747', '#004040']
for idx, (label, lts) in enumerate(lt_scenarios.items()):
    configs = [
        EchelonConfig("Distributor", lead_time=lts[0], holding_cost=0.15, backorder_cost=0.60),
        EchelonConfig("Assembly",    lead_time=lts[1], holding_cost=0.12, backorder_cost=0.50),
        EchelonConfig("Foundry",     lead_time=lts[2], holding_cost=0.08, backorder_cost=0.40),
        EchelonConfig("Wafer",       lead_time=lts[3], holding_cost=0.05, backorder_cost=0.30),
    ]
    ch = SerialSupplyChain.from_config(configs)
    r = ch.simulate(demand, fm, fs)
    bwrs = [er.bullwhip_ratio for er in r.echelon_results]
    cum_bwr = np.cumprod(bwrs)
    ax.plot(range(1,5), cum_bwr, marker=markers[idx], label=f'{label} $L$={lts}',
            linewidth=1.2, markersize=4, color=lt_colors[idx])
    print(f"  {label}: BWR={[f'{b:.2f}' for b in bwrs]}, CumBWR_E4={cum_bwr[-1]:.1f}")

ax.set_xlabel('Echelon', fontsize=7, fontfamily='serif')
ax.set_ylabel('Cumulative BWR', fontsize=7, fontfamily='serif')
ax.set_title('Lead Time Impact on Cumulative Bullwhip', fontsize=8, fontfamily='serif', fontweight='bold')
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(['E1','E2','E3','E4'], fontsize=6)
ax.tick_params(labelsize=6)
ax.legend(fontsize=5, loc='upper left')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
fig_lt.tight_layout()
fig_lt.savefig("figures/fig_leadtime_sensitivity.pdf", dpi=300)
fig_lt.savefig("figures/fig_leadtime_sensitivity.png", dpi=300)
print("Saved: fig_leadtime_sensitivity")

# ── 6. Cost ratio sensitivity ─────────────────────────────────────────
print("\n=== Cost Ratio Sensitivity ===")
cost_ratios = [2.0, 4.0, 6.0, 8.0, 10.0]
total_costs_by_ratio = []
for cr in cost_ratios:
    h = 1.0/(1+cr); b = cr/(1+cr)
    configs = [
        EchelonConfig("Distributor", lead_time=2, holding_cost=h, backorder_cost=b),
        EchelonConfig("Assembly",    lead_time=4, holding_cost=h, backorder_cost=b),
        EchelonConfig("Foundry",     lead_time=12, holding_cost=h, backorder_cost=b),
        EchelonConfig("Wafer",       lead_time=8, holding_cost=h, backorder_cost=b),
    ]
    ch = SerialSupplyChain.from_config(configs)
    r = ch.simulate(demand, fm, fs)
    tc = sum(er.total_cost for er in r.echelon_results)
    total_costs_by_ratio.append(tc)
    print(f"  b/h={cr:.0f}: Total cost={tc:,.1f}")

fig_cr, ax = plt.subplots(figsize=(3.5, 2.5))
ax.bar(range(len(cost_ratios)), total_costs_by_ratio, color='#006747', alpha=0.8, edgecolor='white')
ax.set_xticks(range(len(cost_ratios)))
ax.set_xticklabels([f'{cr:.0f}' for cr in cost_ratios], fontsize=6)
ax.set_xlabel('Cost Ratio $b/h$', fontsize=7, fontfamily='serif')
ax.set_ylabel('Total Chain Cost', fontsize=7, fontfamily='serif')
ax.set_title('Total Supply Chain Cost vs. Cost Asymmetry', fontsize=8, fontfamily='serif', fontweight='bold')
ax.tick_params(labelsize=6)
fig_cr.tight_layout()
fig_cr.savefig("figures/fig_cost_ratio_sensitivity.pdf", dpi=300)
fig_cr.savefig("figures/fig_cost_ratio_sensitivity.png", dpi=300)
print("Saved: fig_cost_ratio_sensitivity")
plt.close('all')

print("\n=== ALL FIGURES GENERATED ===")
import os
for f in sorted(os.listdir("figures")):
    if f.endswith('.png'):
        print(f"  {f}")
