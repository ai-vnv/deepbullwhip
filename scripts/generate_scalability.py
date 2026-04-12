"""Scalability experiments for DeepBullwhip white paper."""
import time
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

gen = SemiconductorDemandGenerator()

# ── A. Scalability in number of Monte Carlo paths N ───────────────────
print("=== A. Scaling with N (Monte Carlo paths) ===")
print(f"{'N':>8s}  {'Serial (s)':>12s}  {'Vectorized (s)':>15s}  {'Speedup':>8s}")

N_values = [10, 50, 100, 500, 1000, 2000, 5000]
serial_times_N = []
vec_times_N = []

for N in N_values:
    demand_batch = gen.generate_batch(T=156, n_paths=N, seed=42)
    fm = np.full_like(demand_batch, demand_batch.mean())
    fs = np.full_like(demand_batch, demand_batch.std())
    
    # Serial: simulate each path individually
    chain = SerialSupplyChain()
    t0 = time.perf_counter()
    for p in range(min(N, 100)):  # cap serial at 100 to avoid timeout
        d = demand_batch[p]
        chain.simulate(d, fm[p], fs[p])
    t_serial = (time.perf_counter() - t0) * (N / min(N, 100))  # extrapolate
    serial_times_N.append(t_serial)
    
    # Vectorized
    vchain = VectorizedSupplyChain()
    t0 = time.perf_counter()
    vchain.simulate(demand_batch, fm, fs)
    t_vec = time.perf_counter() - t0
    vec_times_N.append(t_vec)
    
    speedup = t_serial / t_vec if t_vec > 0 else float('inf')
    print(f"{N:>8d}  {t_serial:>12.3f}  {t_vec:>15.4f}  {speedup:>8.1f}x")

# ── B. Scalability in time horizon T ──────────────────────────────────
print("\n=== B. Scaling with T (time horizon) ===")
print(f"{'T':>8s}  {'Serial (s)':>12s}  {'Vectorized (s)':>15s}  {'Speedup':>8s}")

T_values = [52, 156, 520, 1040, 2080, 5200]
serial_times_T = []
vec_times_T = []
N_fixed = 500

for T in T_values:
    demand_batch = gen.generate_batch(T=T, n_paths=N_fixed, seed=42)
    fm = np.full_like(demand_batch, demand_batch.mean())
    fs = np.full_like(demand_batch, demand_batch.std())
    
    # Serial (sample 20 paths and extrapolate)
    chain = SerialSupplyChain()
    n_sample = min(20, N_fixed)
    t0 = time.perf_counter()
    for p in range(n_sample):
        chain.simulate(demand_batch[p], fm[p], fs[p])
    t_serial = (time.perf_counter() - t0) * (N_fixed / n_sample)
    serial_times_T.append(t_serial)
    
    # Vectorized
    vchain = VectorizedSupplyChain()
    t0 = time.perf_counter()
    vchain.simulate(demand_batch, fm, fs)
    t_vec = time.perf_counter() - t0
    vec_times_T.append(t_vec)
    
    speedup = t_serial / t_vec if t_vec > 0 else float('inf')
    weeks_desc = f"{T//52}yr" if T >= 52 else f"{T}wk"
    print(f"{T:>8d}  {t_serial:>12.3f}  {t_vec:>15.4f}  {speedup:>8.1f}x  ({weeks_desc})")

# ── C. Scalability in number of echelons K ────────────────────────────
print("\n=== C. Scaling with K (echelons) ===")
print(f"{'K':>8s}  {'Serial (s)':>12s}  {'Vectorized (s)':>15s}  {'Speedup':>8s}  {'CumBWR_K':>12s}")

K_values = [2, 4, 6, 8, 10, 12, 16]
serial_times_K = []
vec_times_K = []
cum_bwr_K = []
N_fixed_K = 500
T_fixed = 156

for K in K_values:
    # Build K-echelon chain with reasonable lead times
    configs = []
    for k in range(K):
        lt = max(1, 2 + k)  # increasing lead times: 2,3,4,...
        h = 0.15 - 0.01 * k  # decreasing holding cost
        h = max(h, 0.02)
        b = 0.60 - 0.03 * k  # decreasing backorder cost
        b = max(b, 0.10)
        configs.append(EchelonConfig(f"E{k+1}", lead_time=lt, holding_cost=h, backorder_cost=b))
    
    demand_batch = gen.generate_batch(T=T_fixed, n_paths=N_fixed_K, seed=42)
    fm = np.full_like(demand_batch, demand_batch.mean())
    fs = np.full_like(demand_batch, demand_batch.std())
    
    # Serial (sample)
    chain = SerialSupplyChain.from_config(configs)
    n_sample = min(20, N_fixed_K)
    t0 = time.perf_counter()
    for p in range(n_sample):
        chain.simulate(demand_batch[p], fm[p], fs[p])
    t_serial = (time.perf_counter() - t0) * (N_fixed_K / n_sample)
    serial_times_K.append(t_serial)
    
    # Vectorized
    vchain = VectorizedSupplyChain(configs)
    t0 = time.perf_counter()
    mc_r = vchain.simulate(demand_batch, fm, fs)
    t_vec = time.perf_counter() - t0
    vec_times_K.append(t_vec)
    
    # Get cumulative BWR for one path
    sr = mc_r.to_simulation_result(path_index=0)
    cbwr = np.prod([er.bullwhip_ratio for er in sr.echelon_results])
    cum_bwr_K.append(cbwr)
    
    speedup = t_serial / t_vec if t_vec > 0 else float('inf')
    print(f"{K:>8d}  {t_serial:>12.3f}  {t_vec:>15.4f}  {speedup:>8.1f}x  {cbwr:>12.1f}")

# ── D. Combined scalability stress test ───────────────────────────────
print("\n=== D. Stress Test: Large-scale simulation ===")
stress_configs = [
    (1000, 520, 4, "1K paths, 10yr, 4 echelons"),
    (5000, 156, 4, "5K paths, 3yr, 4 echelons"),
    (1000, 156, 12, "1K paths, 3yr, 12 echelons"),
    (5000, 520, 8, "5K paths, 10yr, 8 echelons"),
    (10000, 156, 4, "10K paths, 3yr, 4 echelons"),
]

stress_results = []
for N, T, K, desc in stress_configs:
    configs = []
    for k in range(K):
        lt = max(1, 2 + k)
        h = max(0.02, 0.15 - 0.01*k)
        b = max(0.10, 0.60 - 0.03*k)
        configs.append(EchelonConfig(f"E{k+1}", lead_time=lt, holding_cost=h, backorder_cost=b))
    
    demand_batch = gen.generate_batch(T=T, n_paths=N, seed=42)
    fm = np.full_like(demand_batch, demand_batch.mean())
    fs = np.full_like(demand_batch, demand_batch.std())
    
    vchain = VectorizedSupplyChain(configs)
    t0 = time.perf_counter()
    vchain.simulate(demand_batch, fm, fs)
    t_vec = time.perf_counter() - t0
    
    mem_mb = (N * K * T * 8 * 3) / (1024**2)  # 3 arrays (orders, inv, cost) of float64
    stress_results.append((desc, N, T, K, t_vec, mem_mb))
    print(f"  {desc:40s}  time={t_vec:.3f}s  mem≈{mem_mb:.0f}MB  ({N*K*T:,.0f} cells)")

# ── E. Generate scalability figures ───────────────────────────────────
plt.rcParams.update({'font.family': 'serif', 'font.size': 8})

# Figure 1: N scaling (log-log)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

ax = axes[0]
ax.loglog(N_values, serial_times_N, 'o-', color='#c4a35a', linewidth=1.2, markersize=4, label='Serial')
ax.loglog(N_values, vec_times_N, 's-', color='#006747', linewidth=1.2, markersize=4, label='Vectorized')
ax.set_xlabel('$N$ (Monte Carlo paths)', fontsize=7)
ax.set_ylabel('Wall time (s)', fontsize=7)
ax.set_title('(a) Scaling with $N$', fontsize=8, fontweight='bold')
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3, which='both')
ax.tick_params(labelsize=6)

# Figure 2: T scaling
ax = axes[1]
ax.loglog(T_values, serial_times_T, 'o-', color='#c4a35a', linewidth=1.2, markersize=4, label='Serial')
ax.loglog(T_values, vec_times_T, 's-', color='#006747', linewidth=1.2, markersize=4, label='Vectorized')
ax.set_xlabel('$T$ (time periods)', fontsize=7)
ax.set_ylabel('Wall time (s)', fontsize=7)
ax.set_title('(b) Scaling with $T$', fontsize=8, fontweight='bold')
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3, which='both')
ax.tick_params(labelsize=6)

# Figure 3: K scaling
ax = axes[2]
ax.semilogy(K_values, serial_times_K, 'o-', color='#c4a35a', linewidth=1.2, markersize=4, label='Serial')
ax.semilogy(K_values, vec_times_K, 's-', color='#006747', linewidth=1.2, markersize=4, label='Vectorized')
ax.set_xlabel('$K$ (echelons)', fontsize=7)
ax.set_ylabel('Wall time (s)', fontsize=7)
ax.set_title('(c) Scaling with $K$', fontsize=8, fontweight='bold')
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3, which='both')
ax.tick_params(labelsize=6)

fig.suptitle('Computational Scalability: Serial vs. Vectorized Engine', fontsize=9, fontweight='bold')
fig.tight_layout()
fig.savefig("figures/fig_scalability.pdf", dpi=300)
fig.savefig("figures/fig_scalability.png", dpi=300)
print("\nSaved: fig_scalability")

# Figure 2: Speedup factor
fig2, ax = plt.subplots(figsize=(3.5, 2.5))
speedups_N = [s/v for s,v in zip(serial_times_N, vec_times_N)]
ax.semilogx(N_values, speedups_N, 'o-', color='#006747', linewidth=1.5, markersize=5)
ax.set_xlabel('$N$ (Monte Carlo paths)', fontsize=7)
ax.set_ylabel('Speedup factor ($\\times$)', fontsize=7)
ax.set_title('Vectorized Speedup vs. Monte Carlo Paths', fontsize=8, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=6)
ax.axhline(y=100, color='#c4a35a', linestyle='--', linewidth=0.8, alpha=0.7, label='$100\\times$ reference')
ax.legend(fontsize=6)
fig2.tight_layout()
fig2.savefig("figures/fig_speedup_vs_N.pdf", dpi=300)
fig2.savefig("figures/fig_speedup_vs_N.png", dpi=300)
print("Saved: fig_speedup_vs_N")

# Figure 3: Cumulative BWR explosion with K
fig3, ax = plt.subplots(figsize=(3.5, 2.5))
ax.semilogy(K_values, cum_bwr_K, 'o-', color='#004040', linewidth=1.5, markersize=5)
ax.set_xlabel('$K$ (number of echelons)', fontsize=7)
ax.set_ylabel('Cumulative BWR at echelon $K$', fontsize=7)
ax.set_title('Bullwhip Amplification vs. Chain Depth', fontsize=8, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.tick_params(labelsize=6)
fig3.tight_layout()
fig3.savefig("figures/fig_bwr_vs_depth.pdf", dpi=300)
fig3.savefig("figures/fig_bwr_vs_depth.png", dpi=300)
print("Saved: fig_bwr_vs_depth")

plt.close('all')
print("\n=== ALL SCALABILITY EXPERIMENTS COMPLETE ===")
