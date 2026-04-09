import numpy as np

from deepbullwhip._types import TimeSeries


def bullwhip_ratio(orders: TimeSeries, demand: TimeSeries) -> float:
    """Variance ratio: Var(orders) / Var(demand)."""
    vd = np.var(demand)
    return float(np.var(orders) / vd) if vd > 0 else 1.0


def fill_rate(inventory_levels: TimeSeries) -> float:
    """Fraction of periods with non-negative inventory."""
    return float(np.mean(np.asarray(inventory_levels) >= 0))


def cumulative_bullwhip(echelon_bw_ratios: list[float]) -> float:
    """Product of per-echelon bullwhip ratios."""
    result = 1.0
    for bw in echelon_bw_ratios:
        result *= bw
    return result


def bullwhip_lower_bound(
    lead_time: int, sensitivity: float, phi: float
) -> float:
    """Theorem 1 lower bound: 1 + 2*L*lam*phi/(1+phi^2) + L^2*lam^2."""
    return (
        1
        + 2 * lead_time * sensitivity * phi / (1 + phi**2)
        + lead_time**2 * sensitivity**2
    )
