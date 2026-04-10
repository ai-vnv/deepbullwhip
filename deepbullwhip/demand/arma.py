"""General ARMA(p,q) demand process generator."""

from __future__ import annotations

import numpy as np

from deepbullwhip._types import TimeSeries
from deepbullwhip.demand.base import DemandGenerator
from deepbullwhip.registry import register


@register("demand", "arma")
class ARMADemandGenerator(DemandGenerator):
    """General ARMA(p,q) demand process.

    Parameters
    ----------
    ar_coeffs : list[float]
        AR coefficients [phi_1, ..., phi_p].
    ma_coeffs : list[float]
        MA coefficients [theta_1, ..., theta_q].
    mu : float
        Long-run mean demand.
    sigma : float
        Innovation standard deviation.
    """

    def __init__(
        self,
        ar_coeffs: list[float] | None = None,
        ma_coeffs: list[float] | None = None,
        mu: float = 12.5,
        sigma: float = 1.0,
    ) -> None:
        self.ar_coeffs = ar_coeffs or [0.7]
        self.ma_coeffs = ma_coeffs or []
        self.mu = mu
        self.sigma = sigma

    def generate(self, T: int, seed: int | None = None) -> TimeSeries:
        rng = np.random.default_rng(seed)
        p = len(self.ar_coeffs)
        q = len(self.ma_coeffs)

        eps = rng.normal(0, self.sigma, T)
        y = np.zeros(T)
        y[:p] = self.mu

        for t in range(p, T):
            ar_part = sum(
                self.ar_coeffs[i] * (y[t - 1 - i] - self.mu) for i in range(p)
            )
            ma_part = sum(
                self.ma_coeffs[j] * eps[t - 1 - j]
                for j in range(min(q, t))
            )
            y[t] = max(0.1, self.mu + ar_part + ma_part + eps[t])

        return y

    def generate_batch(
        self, T: int = 156, n_paths: int = 100, seed: int | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        paths = np.zeros((n_paths, T))
        for i in range(n_paths):
            child_seed = rng.integers(0, 2**31)
            paths[i] = self.generate(T, seed=int(child_seed))
        return paths
