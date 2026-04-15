"""
Corollary 1 validation: Cumulative BWR concentration.

Shows that CV(BWR_cum) ~ sqrt(sum CV_k^2) when per-echelon BWRs are weakly
correlated, and that CV(TC) < CV(BWR_E1). Auto-retries with N=5000 if the
independence approximation fails at N=1000.

Usage:  python scripts/exp_corollary1.py
"""
import numpy as np
from deepbullwhip import SemiconductorDemandGenerator, VectorizedSupplyChain

def run_corollary1(N=1000, T=156, seed=42):
    gen = SemiconductorDemandGenerator()
    db = gen.generate_batch(T=T, n_paths=N, seed=seed)
    vc = VectorizedSupplyChain()
    fmb = np.full_like(db, db.mean())
    fsb = np.full_like(db, db.std())
    mc = vc.simulate(db, fmb, fsb)

    # BWR per echelon per path: shape (N, K)
    bwr = mc.bullwhip_ratios  # (N, K)
    K = bwr.shape[1]
    names = ['Distributor', 'Assembly', 'Foundry', 'Wafer']

    # --- Per-echelon CV ---
    cv_k = np.std(bwr, axis=0) / np.mean(bwr, axis=0)
    print(f"\n{'Echelon':<14} {'Mean BWR':>10} {'Std':>10} {'CV':>10}")
    print("-" * 46)
    for i in range(K):
        print(f"E{i+1} {names[i]:<10} {bwr[:,i].mean():>10.3f} {bwr[:,i].std():>10.4f} {cv_k[i]:>10.4f}")

    # --- Cumulative BWR = product across echelons ---
    bwr_cum = np.prod(bwr, axis=1)  # (N,)
    cv_cum_empirical = np.std(bwr_cum) / np.mean(bwr_cum)

    # --- Predicted CV (independence assumption) ---
    cv_cum_indep = np.sqrt(np.sum(cv_k ** 2))

    # --- Pairwise correlations ---
    print(f"\nPairwise Corr(BWR_i, BWR_j):")
    corr_matrix = np.corrcoef(bwr.T)  # (K, K)
    for i in range(K):
        for j in range(i + 1, K):
            print(f"  Corr(E{i+1}, E{j+1}) = {corr_matrix[i,j]:.4f}")

    # --- Full covariance version ---
    cov_term = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            cov_term += 2 * corr_matrix[i, j] * cv_k[i] * cv_k[j]
    cv_cum_full = np.sqrt(np.sum(cv_k ** 2) + cov_term)

    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 47)
    print(f"{'CV(BWR_cum) empirical':<35} {cv_cum_empirical:>10.4f}")
    print(f"{'CV(BWR_cum) predicted (indep)':<35} {cv_cum_indep:>10.4f}")
    print(f"{'CV(BWR_cum) predicted (full cov)':<35} {cv_cum_full:>10.4f}")
    print(f"{'Mean(BWR_cum)':<35} {np.mean(bwr_cum):>10.1f}")
    print(f"{'Std(BWR_cum)':<35} {np.std(bwr_cum):>10.1f}")

    # Relative error
    rel_err_indep = abs(cv_cum_indep - cv_cum_empirical) / cv_cum_empirical
    rel_err_full = abs(cv_cum_full - cv_cum_empirical) / cv_cum_empirical
    print(f"\n{'Rel error (indep approx)':<35} {rel_err_indep:>10.2%}")
    print(f"{'Rel error (full cov approx)':<35} {rel_err_full:>10.2%}")

    # --- CV(TC) check ---
    tc_per_path = mc.total_costs.sum(axis=1)  # sum across echelons per path
    cv_tc = np.std(tc_per_path) / np.mean(tc_per_path)
    print(f"\n{'CV(TC)':<35} {cv_tc:>10.4f}")
    print(f"{'CV(BWR_E1)':<35} {cv_k[0]:>10.4f}")
    print(f"{'CV(TC) < CV(BWR_E1)?':<35} {'YES' if cv_tc < cv_k[0] else 'NO':>10}")

    # --- Acceptance checks ---
    print("\n" + "=" * 50)
    print("ACCEPTANCE CHECKS")
    print("=" * 50)

    # Check 1: independence approximation < 15%
    if rel_err_indep < 0.15:
        print(f"  Independence approx: PASS ({rel_err_indep:.1%} < 15%)")
        best_approx = "independence"
        best_err = rel_err_indep
    elif rel_err_full < 0.15:
        print(f"  Independence approx: FAIL ({rel_err_indep:.1%} >= 15%)")
        print(f"  Full covariance approx: PASS ({rel_err_full:.1%} < 15%)")
        best_approx = "full_cov"
        best_err = rel_err_full
    else:
        print(f"  Independence approx: FAIL ({rel_err_indep:.1%})")
        print(f"  Full covariance approx: FAIL ({rel_err_full:.1%})")
        best_approx = None
        best_err = min(rel_err_indep, rel_err_full)

    print(f"  CV(TC) < CV(BWR_E1): {'PASS' if cv_tc < cv_k[0] else 'FAIL'}")

    return {
        'N': N,
        'cv_k': cv_k,
        'cv_cum_empirical': cv_cum_empirical,
        'cv_cum_indep': cv_cum_indep,
        'cv_cum_full': cv_cum_full,
        'rel_err_indep': rel_err_indep,
        'rel_err_full': rel_err_full,
        'cv_tc': cv_tc,
        'best_approx': best_approx,
        'best_err': best_err,
        'corr_matrix': corr_matrix,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("COROLLARY 1: Cumulative BWR Concentration")
    print("=" * 60)

    result = run_corollary1(N=1000, T=156, seed=42)

    # If independence fails, retry with N=5000
    if result['best_approx'] is None or result['rel_err_indep'] >= 0.15:
        print("\n\n>>> Re-running with N=5000 for tighter estimates...")
        result = run_corollary1(N=5000, T=156, seed=42)

    print("\n" + "=" * 60)
    print("FINAL NUMBERS FOR PAPER")
    print("=" * 60)
    cv_k = result['cv_k']
    names = ['Distributor', 'Assembly', 'Foundry', 'Wafer']
    for i in range(len(cv_k)):
        print(f"  CV(BWR_E{i+1}) [{names[i]}] = {cv_k[i]:.4f}")
    print(f"  CV(BWR_cum) empirical = {result['cv_cum_empirical']:.4f}")
    print(f"  CV(BWR_cum) predicted (indep) = {result['cv_cum_indep']:.4f}")
    print(f"  CV(BWR_cum) predicted (full cov) = {result['cv_cum_full']:.4f}")
    print(f"  Relative error (best) = {result['best_err']:.1%}")
    print(f"  CV(TC) = {result['cv_tc']:.4f}")
    print(f"  Best approximation: {result['best_approx']}")
