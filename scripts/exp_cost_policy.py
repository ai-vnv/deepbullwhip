"""
Cost asymmetry x Policy interaction experiment.

Tests 3 policies (OUT, POUT, Smoothing) x 4 cost ratios (b/h = 2, 5, 10, 20)
to detect ranking swaps under different cost asymmetries.

Usage:  python scripts/exp_cost_policy.py
"""
import numpy as np
from deepbullwhip import EchelonConfig
from deepbullwhip.benchmark import BenchmarkRunner

def run_cost_policy(N=500, T=156, seed=42):
    bh_ratios = [2, 5, 10, 20]
    policies = [
        'order_up_to',
        ('proportional_out', {'alpha': 0.5}),
        'smoothing_out',
    ]
    policy_labels = ['OUT', 'POUT(α=0.5)', 'Smoothing']
    lead_times = [2, 4, 12, 8]

    results = {}

    for bh in bh_ratios:
        h = 1.0 / (1.0 + bh)
        b = bh / (1.0 + bh)

        configs = [
            EchelonConfig(f'E{k+1}', lead_time=lead_times[k],
                          holding_cost=h, backorder_cost=b,
                          service_level=0.95)
            for k in range(4)
        ]

        runner = BenchmarkRunner(
            chain_config=configs,
            demand='semiconductor_ar1',
            T=T, N=N, seed=seed
        )

        df = runner.run(policies=policies, metrics=['TC', 'CUM_BWR', 'FILL_RATE'])

        for i, (pol_spec, label) in enumerate(zip(policies, policy_labels)):
            pol_name = pol_spec if isinstance(pol_spec, str) else pol_spec[0]
            sub = df[df['policy'] == pol_name]
            tc_vals = sub[sub['metric'] == 'TC']
            tc_total = tc_vals['value'].sum()
            tc_e4 = tc_vals[tc_vals['echelon'] == 'E4']['value'].values[0]
            cum_bwr = sub[(sub['metric'] == 'CUM_BWR') & (sub['echelon'] == 'E4')]['value'].values[0]
            fr = sub[(sub['metric'] == 'FILL_RATE') & (sub['echelon'] == 'E4')]['value'].values[0]

            results[(bh, label)] = {
                'TC_total': tc_total, 'TC_E4': tc_e4,
                'CUM_BWR': cum_bwr, 'FR': fr,
                'h': h, 'b': b,
            }
            print(f"  b/h={bh:>2d}  {label:<14s}  TC={tc_total:>10.1f}  TC_E4={tc_e4:>10.1f}  "
                  f"CUM_BWR={cum_bwr:>8.1f}  FR={fr:.1%}")

    return results, bh_ratios, policy_labels


if __name__ == '__main__':
    print("=" * 60)
    print("COST ASYMMETRY × POLICY INTERACTION")
    print("=" * 60)

    results, bh_ratios, policy_names = run_cost_policy(N=500, T=156, seed=42)

    # Print table: TC_total
    print(f"\nTotal Cost (all echelons, mean over paths):")
    print(f"{'b/h':>6}", end='')
    for pn in policy_names:
        print(f"  {pn:>14}", end='')
    print()
    print("-" * (6 + 16 * len(policy_names)))

    for bh in bh_ratios:
        print(f"{bh:>6}", end='')
        for pn in policy_names:
            tc = results[(bh, pn)]['TC_total']
            print(f"  {tc:>14.1f}", end='')
        print()

    # Check for ranking swap
    print("\n" + "=" * 50)
    print("ACCEPTANCE: Ranking swap detection")
    print("=" * 50)

    found_swap = False
    for i, bh1 in enumerate(bh_ratios):
        for bh2 in bh_ratios[i+1:]:
            rank1 = sorted(policy_names, key=lambda pn: results[(bh1, pn)]['TC_total'])
            rank2 = sorted(policy_names, key=lambda pn: results[(bh2, pn)]['TC_total'])
            if rank1 != rank2:
                found_swap = True
                print(f"  SWAP found between b/h={bh1} and b/h={bh2}:")
                print(f"    b/h={bh1}: {' < '.join(rank1)}")
                print(f"    b/h={bh2}: {' < '.join(rank2)}")

    print(f"\n  Ranking swap: {'PASS' if found_swap else 'FAIL'}")

    print("\n" + "=" * 60)
    print("FINAL NUMBERS FOR PAPER")
    print("=" * 60)
    print(f"{'b/h':>6} {'Policy':>14} {'TC_total':>12} {'TC_E4':>12} {'CUM_BWR':>10} {'FR':>8}")
    for bh in bh_ratios:
        for pn in policy_names:
            r = results[(bh, pn)]
            print(f"{bh:>6} {pn:>14} {r['TC_total']:>12.1f} {r['TC_E4']:>12.1f} {r['CUM_BWR']:>10.1f} {r['FR']:>8.1%}")
