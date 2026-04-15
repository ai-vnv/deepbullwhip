"""
Reproduce all experimental results from the CAIE 2026 paper.

Paper:  Arief (2026), "Accuracy-Robustness Tradeoffs in ML-Driven
        Semiconductor Supply Chains", Computers & Industrial Engineering.

Usage:  python scripts/run_caie_experiments.py

Experiments:
  1. Single-path bullwhip propagation
  2. Monte Carlo stochastic filtering (N=1000)
  3. Lead time sensitivity (3 scenarios)
  4. Scalability benchmarks (serial vs vectorized)
  5a. Policy comparison (OUT, POUT, Smoothing, Constant)
  5b. Forecaster comparison (Naive, MA, SES, DeepAR)
  5c. Cross-chain comparison (Semiconductor, Beer Game, Consumer)
  5d. Real WSTS vs synthetic demand
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from deepbullwhip import (
    SemiconductorDemandGenerator, SerialSupplyChain,
    VectorizedSupplyChain, EchelonConfig,
)
from deepbullwhip.benchmark import BenchmarkRunner
from deepbullwhip.datasets import load_wsts
from deepbullwhip.demand.replay import ReplayDemandGenerator
import time

gen = SemiconductorDemandGenerator()

print("=" * 70)
print("EXPERIMENT 1: Single-path bullwhip propagation")
print("=" * 70)
d = gen.generate(T=156, seed=42)
chain = SerialSupplyChain()
fm, fs = np.full_like(d, d.mean()), np.full_like(d, d.std())
r = chain.simulate(d, fm, fs)
cum = 1.0
print(f"{'Ech':<4} {'Role':<14} {'BWR':>8} {'FR':>8} {'Cost':>8} {'CumBWR':>10}")
for k, er in enumerate(r.echelon_results):
    cum *= er.bullwhip_ratio
    print(f"E{k+1:<3} {er.name:<14} {er.bullwhip_ratio:>8.2f} {er.fill_rate:>8.1%} "
          f"{er.total_cost:>8.0f} {cum:>10.1f}")
print(f"{'':4} {'Total':<14} {'---':>8} {'---':>8} {r.total_cost:>8.0f} {cum:>10.1f}")

print("\n" + "=" * 70)
print("EXPERIMENT 2: Monte Carlo stochastic filtering (N=1000)")
print("=" * 70)
db = gen.generate_batch(T=156, n_paths=1000, seed=42)
vc = VectorizedSupplyChain()
fmb, fsb = np.full_like(db, db.mean()), np.full_like(db, db.std())
mc = vc.simulate(db, fmb, fsb)
bwr = np.array([[er.bullwhip_ratio for er in mc.to_simulation_result(p).echelon_results] for p in range(1000)])
names = ['Distributor', 'Assembly', 'Foundry', 'Wafer']
print(f"{'Ech':<4} {'Role':<12} {'Mean':>8} {'Median':>8} {'Std':>8} {'CV':>8}")
for i, n in enumerate(names):
    m, med, s = bwr[:,i].mean(), np.median(bwr[:,i]), bwr[:,i].std()
    print(f"E{i+1:<3} {n:<12} {m:>8.1f} {med:>8.1f} {s:>8.2f} {s/m:>8.3f}")

print("\n" + "=" * 70)
print("EXPERIMENT 3: Lead time sensitivity")
print("=" * 70)
for label, lts in [("Short", [1,2,4,2]), ("Baseline", [2,4,12,8]), ("Long", [4,8,20,12])]:
    configs = [EchelonConfig(f'E{k+1}', lead_time=lts[k],
               holding_cost=[.15,.12,.08,.05][k],
               backorder_cost=[.60,.50,.40,.30][k]) for k in range(4)]
    ch = SerialSupplyChain.from_config(configs)
    res = ch.simulate(d, fm, fs)
    cum = np.prod([er.bullwhip_ratio for er in res.echelon_results])
    print(f"  {label:10s} L={lts}  CumBWR = {cum:.1f}")

print("\n" + "=" * 70)
print("EXPERIMENT 4: Scalability benchmarks")
print("=" * 70)
for N in [100, 500, 1000, 5000]:
    db = gen.generate_batch(T=156, n_paths=N, seed=42)
    fmb, fsb = np.full_like(db, db.mean()), np.full_like(db, db.std())
    ch = SerialSupplyChain(); ns = min(20, N)
    t0 = time.perf_counter()
    for p in range(ns): ch.simulate(db[p], fmb[p], fsb[p])
    t_ser = (time.perf_counter()-t0)*(N/ns)
    v = VectorizedSupplyChain()
    t0 = time.perf_counter(); v.simulate(db, fmb, fsb); t_vec = time.perf_counter()-t0
    print(f"  N={N:>5d}: serial={t_ser:.2f}s, vec={t_vec:.4f}s, speedup={t_ser/t_vec:.0f}x")

print("\n" + "=" * 70)
print("EXPERIMENT 5a: Policy comparison (N=1000)")
print("=" * 70)
runner = BenchmarkRunner('semiconductor_4tier','semiconductor_ar1',T=156,N=1000,seed=42)
r5a = runner.run(
    policies=['order_up_to',('proportional_out',{'alpha':0.3}),'smoothing_out','constant_order'],
    metrics=['BWR','CUM_BWR','TC','FILL_RATE','NSAmp'])
print(r5a.pivot_table(index=['policy','echelon'],columns='metric',values='value').to_string(float_format='%.1f'))

print("\n" + "=" * 70)
print("EXPERIMENT 5b: Forecaster comparison (N=1000)")
print("=" * 70)

# --- Train DeepAR on synthetic demand corpus ---
try:
    from deepbullwhip.forecast.deepar import DeepARTrainer
    print("  Training DeepAR on 200 synthetic demand paths...")
    train_series = [gen.generate(T=260, seed=s) for s in range(200)]
    trainer = DeepARTrainer(
        freq="W", prediction_length=1, context_length=52,
        epochs=30, num_layers=3, hidden_size=40,
    )
    deepar_fc = trainer.train(train_series)
    # Inject trained instance into registry for BenchmarkRunner
    from deepbullwhip.registry import _REGISTRY
    _REGISTRY["forecaster"]["deepar"] = lambda **kw: deepar_fc
    deepar_available = True
    print("  DeepAR training complete.")
except ImportError:
    print("  [SKIP] GluonTS not installed — skipping DeepAR.")
    deepar_available = False
except Exception as e:
    print(f"  [SKIP] DeepAR training failed: {e}")
    deepar_available = False

forecaster_list = [
    'naive',
    ('moving_average', {'window': 10}),
    ('exponential_smoothing', {'alpha': 0.3}),
]
if deepar_available:
    forecaster_list.append('deepar')

r5b = runner.run(
    policies=['order_up_to'],
    forecasters=forecaster_list,
    metrics=['BWR', 'CUM_BWR', 'TC', 'FILL_RATE'])
print(r5b.pivot_table(index=['forecaster','echelon'],
      columns='metric', values='value').to_string(float_format='%.1f'))

print("\n" + "=" * 70)
print("EXPERIMENT 5c: Cross-chain comparison (N=500)")
print("=" * 70)
for cn,dn,T in [('semiconductor_4tier','semiconductor_ar1',156),
                ('beer_game','beer_game',52),
                ('consumer_2tier','semiconductor_ar1',156)]:
    r3 = BenchmarkRunner(cn,dn,T=T,N=500,seed=42)
    res = r3.run(policies=['order_up_to'],metrics=['CUM_BWR','TC'])
    last = res[res['metric']=='CUM_BWR']['echelon'].max()
    cum = res[(res['metric']=='CUM_BWR')&(res['echelon']==last)]['value'].values[0]
    tc = res[res['metric']=='TC']['value'].sum()
    print(f"  {cn:25s} CumBWR={cum:>10.1f}  TC={tc:>8.0f}")

print("\n" + "=" * 70)
print("EXPERIMENT 5d: Real WSTS vs synthetic (N=500)")
print("=" * 70)
wsts = load_wsts()
replay = ReplayDemandGenerator(data=wsts)
r4 = BenchmarkRunner('semiconductor_4tier',replay,T=len(wsts),N=500,seed=42)
r5d = r4.run(policies=['order_up_to',('proportional_out',{'alpha':0.5})],metrics=['CUM_BWR','TC'])
print(r5d.pivot_table(index=['policy','echelon'],columns='metric',values='value').to_string(float_format='%.1f'))

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
