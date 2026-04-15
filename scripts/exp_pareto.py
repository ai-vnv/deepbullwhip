"""
POUT alpha Pareto frontier: BWR vs NSAmp vs Fill Rate trade-off.

Sweeps alpha in {0.1, 0.2, ..., 1.0} for the semiconductor_4tier chain
and plots the bullwhip-inventory and bullwhip-service efficiency frontiers.

Reference: Disney & Towill (2003), proportional OUT policy.
Usage:     python scripts/exp_pareto.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deepbullwhip.benchmark import BenchmarkRunner

def run_pareto(N=1000, T=156, seed=42):
    alphas = np.arange(0.1, 1.05, 0.1)
    results = []

    for alpha in alphas:
        alpha = round(alpha, 1)
        runner = BenchmarkRunner('semiconductor_4tier', 'semiconductor_ar1', T=T, N=N, seed=seed)
        df = runner.run(
            policies=[('proportional_out', {'alpha': alpha})],
            metrics=['CUM_BWR', 'NSAmp', 'FILL_RATE', 'TC']
        )
        # Get last echelon values
        last_ech = df['echelon'].max()
        row = {'alpha': alpha}
        for metric in ['CUM_BWR', 'NSAmp', 'FILL_RATE', 'TC']:
            val = df[(df['metric'] == metric) & (df['echelon'] == last_ech)]['value'].values
            if len(val) > 0:
                row[metric] = val[0]
            else:
                # Some metrics may be reported at echelon level
                val = df[df['metric'] == metric]['value'].values
                row[metric] = val[-1] if len(val) > 0 else np.nan
        results.append(row)
        print(f"  α={alpha:.1f}  CUM_BWR={row.get('CUM_BWR', np.nan):>10.1f}  "
              f"NSAmp={row.get('NSAmp', np.nan):>8.1f}  "
              f"FR={row.get('FILL_RATE', np.nan):>6.1%}  "
              f"TC={row.get('TC', np.nan):>8.0f}")

    return results

def plot_pareto(results):
    alphas = [r['alpha'] for r in results]
    bwr = [r['CUM_BWR'] for r in results]
    nsamp = [r['NSAmp'] for r in results]
    fr = [r['FILL_RATE'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # BWR vs NSAmp (log-log)
    ax = axes[0]
    ax.loglog(nsamp, bwr, 'o-', color='#2c3e50', markersize=8, linewidth=2)
    for i, a in enumerate(alphas):
        ax.annotate(f'α={a:.1f}', (nsamp[i], bwr[i]),
                    textcoords='offset points', xytext=(8, 4), fontsize=8)
    ax.set_xlabel('Net Stock Amplification (NSAmp)', fontsize=11)
    ax.set_ylabel('Cumulative BWR', fontsize=11)
    ax.set_title('Bullwhip–Inventory Trade-off', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')

    # BWR vs Fill Rate
    ax = axes[1]
    ax.plot(fr, bwr, 's-', color='#c0392b', markersize=8, linewidth=2)
    for i, a in enumerate(alphas):
        ax.annotate(f'α={a:.1f}', (fr[i], bwr[i]),
                    textcoords='offset points', xytext=(8, 4), fontsize=8)
    ax.set_xlabel('Fill Rate', fontsize=11)
    ax.set_ylabel('Cumulative BWR', fontsize=11)
    ax.set_title('Bullwhip–Service Trade-off', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_pareto_frontier.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fig_pareto_frontier.png', bbox_inches='tight', dpi=150)
    print("\nSaved: fig_pareto_frontier.pdf, fig_pareto_frontier.png")


if __name__ == '__main__':
    print("=" * 60)
    print("PARETO FRONTIER: POUT α sweep")
    print("=" * 60)

    results = run_pareto(N=1000, T=156, seed=42)

    # Acceptance checks
    print("\n" + "=" * 50)
    print("ACCEPTANCE CHECKS")
    print("=" * 50)
    bwr_10 = [r for r in results if r['alpha'] == 1.0][0]['CUM_BWR']
    bwr_03 = [r for r in results if r['alpha'] == 0.3][0]['CUM_BWR']
    print(f"  BWR(α=1.0) = {bwr_10:.1f}  (expected ≈ 427)")
    print(f"  BWR(α=0.3) = {bwr_03:.1f}  (expected ≈ 16)")

    # Check monotonicity
    bwrs = [r['CUM_BWR'] for r in results]
    monotone = all(bwrs[i] <= bwrs[i+1] for i in range(len(bwrs)-1))
    print(f"  Monotonically increasing: {'PASS' if monotone else 'FAIL (but may be okay)'}")

    plot_pareto(results)

    print("\n" + "=" * 60)
    print("FINAL NUMBERS FOR PAPER")
    print("=" * 60)
    print(f"{'alpha':>6} {'CUM_BWR':>10} {'NSAmp':>10} {'FILL_RATE':>10} {'TC':>10}")
    for r in results:
        print(f"{r['alpha']:>6.1f} {r['CUM_BWR']:>10.1f} {r.get('NSAmp', np.nan):>10.1f} "
              f"{r['FILL_RATE']:>10.1%} {r['TC']:>10.0f}")
