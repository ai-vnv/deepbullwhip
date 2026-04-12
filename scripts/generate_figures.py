"""Generate professional schematic, all diagnostic plots, and scalability figures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import time

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
    plot_summary_dashboard,
    plot_echelon_detail,
)

gen = SemiconductorDemandGenerator()
demand = gen.generate(T=156, seed=42)
chain = SerialSupplyChain()
fm = np.full_like(demand, demand.mean())
fs = np.full_like(demand, demand.std())
result = chain.simulate(demand, fm, fs)

# ═══════════════════════════════════════════════════════════════════════
# 1. PROFESSIONAL SEMICONDUCTOR SUPPLY CHAIN SCHEMATIC
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(-0.8, 4.5)
ax.axis('off')

# Semiconductor-specific tier data
tiers = [
    {"name": "Wafer &\nMaterial\nSupplier", "abbr": "E4", "x": 1.5, "lt": 8,
     "detail": "Silicon ingots,\nphotomasks, gases", "color": "#9e8340"},
    {"name": "Foundry\n/ Fab", "abbr": "E3", "x": 5.0, "lt": 12,
     "detail": "Wafer processing,\nlithography, etching", "color": "#004040"},
    {"name": "Assembly\n& Test\n(OSAT)", "abbr": "E2", "x": 8.5, "lt": 4,
     "detail": "Die bonding,\npackaging, test", "color": "#c4a35a"},
    {"name": "Distributor\n/ OEM", "abbr": "E1", "x": 12.0, "lt": 2,
     "detail": "Board assembly,\nfinal product", "color": "#006747"},
]

for t in tiers:
    # Main box with rounded corners
    box = FancyBboxPatch((t["x"]-1.1, 1.0), 2.2, 2.0,
                          boxstyle="round,pad=0.1",
                          facecolor=t["color"], edgecolor='white',
                          linewidth=0, alpha=0.12, zorder=1)
    ax.add_patch(box)
    border = FancyBboxPatch((t["x"]-1.1, 1.0), 2.2, 2.0,
                             boxstyle="round,pad=0.1",
                             facecolor='none', edgecolor=t["color"],
                             linewidth=1.5, zorder=2)
    ax.add_patch(border)

    # Tier label
    ax.text(t["x"], 2.55, t["abbr"], ha='center', va='center',
            fontsize=9, fontfamily='serif', fontweight='bold',
            color=t["color"], zorder=3)
    ax.text(t["x"], 1.85, t["name"], ha='center', va='center',
            fontsize=6.5, fontfamily='serif', color='#333333',
            linespacing=1.1, zorder=3)

    # Lead time badge below box
    badge = FancyBboxPatch((t["x"]-0.55, 0.35), 1.1, 0.45,
                            boxstyle="round,pad=0.05",
                            facecolor=t["color"], edgecolor='none',
                            alpha=0.85, zorder=2)
    ax.add_patch(badge)
    ax.text(t["x"], 0.575, f'$L = {t["lt"]}$ wk', ha='center', va='center',
            fontsize=6, fontfamily='serif', color='white', fontweight='bold', zorder=3)

    # Detail text above box
    ax.text(t["x"], 3.25, t["detail"], ha='center', va='bottom',
            fontsize=5, fontfamily='serif', color='#888888',
            linespacing=1.15, style='italic', zorder=3)

# Arrows between tiers
for i in range(len(tiers)-1):
    x1 = tiers[i]["x"] + 1.15
    x2 = tiers[i+1]["x"] - 1.15
    mid = (x1 + x2) / 2

    # Material flow (downstream) - bottom arrow
    ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                arrowprops=dict(arrowstyle='->', color='#004040',
                               lw=1.8, connectionstyle='arc3,rad=0'))
    ax.text(mid, 1.25, 'Material', ha='center', va='top', fontsize=5,
            color='#004040', fontfamily='serif')

    # Order flow (upstream) - top arrow
    ax.annotate('', xy=(x1, 2.5), xytext=(x2, 2.5),
                arrowprops=dict(arrowstyle='->', color='#c4a35a',
                               lw=1.8, connectionstyle='arc3,rad=0'))
    ax.text(mid, 2.7, 'Orders', ha='center', va='bottom', fontsize=5,
            color='#9e8340', fontfamily='serif')

# Customer demand arrow
ax.annotate('', xy=(tiers[-1]["x"]+1.15, 2.0), xytext=(tiers[-1]["x"]+2.2, 2.0),
            arrowprops=dict(arrowstyle='->', color='#1a1a1a', lw=1.8))

# Customer icon (simple)
cx = tiers[-1]["x"] + 2.7
ax.text(cx, 2.0, 'D(t)', ha='center', va='center', fontsize=10, fontfamily='serif', style='italic', zorder=3)
ax.text(cx, 1.3, 'End\nCustomer', ha='center', va='top', fontsize=6,
        fontfamily='serif', color='#555555')

# Title
ax.text(7.0, 4.2, 'Semiconductor Supply Chain: Four-Echelon Serial Topology',
        ha='center', va='center', fontsize=10, fontfamily='serif',
        fontweight='bold', color='#1a1a1a')

# Variance amplification annotation
ax.annotate('', xy=(1.5, -0.3), xytext=(12.0, -0.3),
            arrowprops=dict(arrowstyle='<->', color='#c4a35a', lw=1.0))
ax.text(6.75, -0.55, 'Bullwhip amplification: BWR$_{\\mathrm{cum}}$ = 837.6×',
        ha='center', va='top', fontsize=7, fontfamily='serif',
        color='#9e8340', style='italic')

fig.tight_layout(pad=0.5)
fig.savefig("figures/fig_supply_chain_schematic.pdf", dpi=300, bbox_inches='tight')
fig.savefig("figures/fig_supply_chain_schematic.png", dpi=300, bbox_inches='tight')
print("Saved: professional schematic")
plt.close('all')

# ═══════════════════════════════════════════════════════════════════════
# 2. ALL DIAGNOSTIC PLOTS FOR APPENDIX
# ═══════════════════════════════════════════════════════════════════════
fig1 = plot_demand_trajectory(demand, width="double")
fig1.savefig("figures/fig_demand_trajectory.pdf", dpi=300)
print("Saved: demand_trajectory")

fig2 = plot_order_quantities(demand, result, width="double")
fig2.savefig("figures/fig_order_quantities.pdf", dpi=300)
print("Saved: order_quantities")

fig3 = plot_inventory_levels(result, width="double")
fig3.savefig("figures/fig_inventory_levels.pdf", dpi=300)
print("Saved: inventory_levels")

fig4 = plot_cost_timeseries(result, width="double")
fig4.savefig("figures/fig_cost_timeseries.pdf", dpi=300)
print("Saved: cost_timeseries")

fig7 = plot_summary_dashboard(demand, result)
fig7.savefig("figures/fig_summary_dashboard.pdf", dpi=300)
print("Saved: summary_dashboard")

for eidx, ename in [(0,"distributor"), (1,"assembly"), (2,"foundry"), (3,"wafer")]:
    fig = plot_echelon_detail(demand, result, echelon_index=eidx, width="double")
    fig.savefig(f"figures/fig_echelon_detail_{ename}.pdf", dpi=300)
    print(f"Saved: echelon_detail_{ename}")

plt.close('all')

# ═══════════════════════════════════════════════════════════════════════
# 3. BENCHMARK: DeepBullwhip vs naive Python loop
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Benchmark: DeepBullwhip engines vs naive Python ===")

def naive_bullwhip_sim(demand, L=2, h=0.15, b=0.60):
    """Naive Python loop simulation (no vectorization, no ABC)."""
    T = len(demand)
    orders = np.zeros(T)
    inventory = np.zeros(T)
    cost = np.zeros(T)
    pipeline = [0.0] * L
    mu_hat = demand.mean()
    sig_hat = demand.std()
    z = 0.84  # ~80th percentile
    for t in range(T):
        ip = inventory[t-1] if t > 0 else 0.0
        ip += sum(pipeline)
        S = mu_hat * (L+1) + z * sig_hat * np.sqrt(L+1)
        orders[t] = max(0, S - ip)
        received = pipeline.pop(0)
        pipeline.append(orders[t])
        inv = (inventory[t-1] if t > 0 else 0.0) + received - demand[t]
        inventory[t] = inv
        cost[t] = h * max(inv, 0) + b * max(-inv, 0)
    return orders, inventory, cost

bench_configs = [
    (100, 156, "100 paths, 3yr"),
    (500, 156, "500 paths, 3yr"),
    (1000, 156, "1K paths, 3yr"),
    (1000, 520, "1K paths, 10yr"),
]

print(f"{'Config':30s} {'Naive (s)':>10s} {'Serial (s)':>10s} {'Vec (s)':>10s} {'Vec/Naive':>10s}")
bench_results = []
for N, T, desc in bench_configs:
    demand_batch = gen.generate_batch(T=T, n_paths=N, seed=42)

    # Naive Python
    t0 = time.perf_counter()
    n_sample = min(20, N)
    for p in range(n_sample):
        naive_bullwhip_sim(demand_batch[p])
    t_naive = (time.perf_counter() - t0) * (N / n_sample)

    # DeepBullwhip Serial
    chain = SerialSupplyChain()
    t0 = time.perf_counter()
    for p in range(n_sample):
        d = demand_batch[p]
        f_m = np.full_like(d, d.mean())
        f_s = np.full_like(d, d.std())
        chain.simulate(d, f_m, f_s)
    t_serial = (time.perf_counter() - t0) * (N / n_sample)

    # DeepBullwhip Vectorized
    vchain = VectorizedSupplyChain()
    f_m = np.full_like(demand_batch, demand_batch.mean())
    f_s = np.full_like(demand_batch, demand_batch.std())
    t0 = time.perf_counter()
    vchain.simulate(demand_batch, f_m, f_s)
    t_vec = time.perf_counter() - t0

    speedup = t_naive / t_vec
    bench_results.append((desc, t_naive, t_serial, t_vec, speedup))
    print(f"{desc:30s} {t_naive:>10.2f} {t_serial:>10.2f} {t_vec:>10.4f} {speedup:>9.0f}x")

print("\n=== All generation complete ===")
