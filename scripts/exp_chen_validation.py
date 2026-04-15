"""
Chen et al. (2000) single-echelon analytical bound validation.

Reference: Chen, Drezner, Ryan & Simchi-Levi (2000), "Quantifying the
    Bullwhip Effect in a Simple Supply Chain", Management Science, 46(3).
    https://doi.org/10.1287/mnsc.46.3.436.12069

Verifies simulated BWR matches the Chen formula for OUT with MA(p) forecast
and i.i.d. Normal demand.

Formula:  BWR = 1 + 2L_r/p + 2L_r^2/p^2
where L_r = lead time + review period (= l + 1 for periodic-review with R=1).

Usage:    python scripts/exp_chen_validation.py
"""
import numpy as np
from scipy import stats

def chen_formula(l, p):
    """Chen et al. (2000) BWR for OUT with MA(p).

    Parameters
    ----------
    l : int
        Replenishment lead time (periods).
    p : int
        Moving average window.

    Returns
    -------
    float
        BWR = 1 + 2(l+1)/p + 2(l+1)²/p² where l+1 = lead time + review period.
    """
    Lr = l + 1
    return 1.0 + 2.0 * Lr / p + 2.0 * Lr**2 / p**2


def simulate_chen_echelon(demand, l, p, z_alpha, sigma):
    """Simulate Chen's OUT model with MA(p) forecast on a single echelon.

    Timing per period:
      1. Receive pipeline order (placed l periods ago)
      2. Demand d_t occurs
      3. Update forecast: fm_t = MA(p) including d_t
      4. Compute S_t = (l+1)*fm_t + z*sigma*sqrt(l+1)
      5. Order = max(0, S_t - IP_t)
    """
    T = len(demand)
    orders = np.zeros(T)
    pipeline = np.zeros(l + 1)

    mu_init = demand[:min(p, T)].mean()
    S0 = (l + 1) * mu_init + z_alpha * sigma * np.sqrt(l + 1)
    inv_current = S0

    for t in range(T):
        received = pipeline[0]
        pipeline = np.roll(pipeline, -1)
        pipeline[-1] = 0.0
        inv_current += received - demand[t]

        if t >= p - 1:
            fm_t = demand[t - p + 1:t + 1].mean()
        else:
            fm_t = demand[:t + 1].mean()

        S_t = (l + 1) * fm_t + z_alpha * sigma * np.sqrt(l + 1)
        IP = inv_current + pipeline.sum()
        order = max(0.0, S_t - IP)
        orders[t] = order
        pipeline[-1] = order

    return orders


def run_chen_validation(N=2000, T=2000, mu=100.0, sigma=10.0, seed=42, warmup=200):
    z_alpha = stats.norm.ppf(0.95)

    test_cases = [
        (2, 10), (4, 10), (4, 20), (8, 20),
        (8, 52), (12, 52), (2, 52), (12, 10),
    ]

    rng = np.random.default_rng(seed)

    print(f"\n{'l':>4} {'p':>4} {'L_r':>4} {'Chen':>10} {'Sim BWR':>10} {'Rel Err':>10} {'Status':>8}")
    print("-" * 54)

    results = []
    all_pass = True

    for l, p in test_cases:
        analytical = chen_formula(l, p)
        bwr_list = []

        for _ in range(N):
            path_seed = int(rng.integers(0, 2**31))
            path_rng = np.random.default_rng(path_seed)
            demand = path_rng.normal(mu, sigma, T + warmup)
            demand = np.maximum(demand, 0.0)

            orders = simulate_chen_echelon(demand, l, p, z_alpha, sigma)

            var_d = np.var(demand[warmup:])
            var_o = np.var(orders[warmup:])
            if var_d > 0:
                bwr_list.append(var_o / var_d)

        sim_bwr = np.mean(bwr_list)
        sim_std = np.std(bwr_list)
        rel_err = (sim_bwr - analytical) / analytical
        abs_err = abs(rel_err)
        status = "PASS" if abs_err < 0.05 else "MARGINAL" if abs_err < 0.10 else "FAIL"
        if abs_err >= 0.10:
            all_pass = False

        print(f"{l:>4d} {p:>4d} {l+1:>4d} {analytical:>10.4f} {sim_bwr:>10.4f} "
              f"{rel_err:>+10.2%} {status:>8}")
        results.append({
            'l': l, 'p': p, 'Lr': l + 1,
            'chen_bound': analytical, 'sim_bwr': sim_bwr,
            'rel_err': rel_err, 'status': status, 'bwr_std': sim_std,
        })

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"N={N}, T={T}, warmup={warmup}, mu={mu}, sigma={sigma}")
    return results, all_pass


if __name__ == '__main__':
    print("=" * 60)
    print("CHEN VALIDATION: Single-echelon BWR vs analytical formula")
    print("=" * 60)
    print("Formula: BWR = 1 + 2(l+1)/p + 2(l+1)²/p²")
    print("  l = replenishment lead time, p = MA window, R=1 (review)")

    results, passed = run_chen_validation(N=2000, T=2000, warmup=200, seed=42)

    print("\n" + "=" * 60)
    print("FINAL NUMBERS FOR PAPER")
    print("=" * 60)
    print(f"{'l':>4} {'p':>4} {'L_r=l+1':>8} {'Chen':>10} {'Sim':>10} {'Err':>8}")
    for r in results:
        print(f"{r['l']:>4d} {r['p']:>4d} {r['Lr']:>8d} {r['chen_bound']:>10.4f} "
              f"{r['sim_bwr']:>10.4f} {r['rel_err']:>+8.2%}")
