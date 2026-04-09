import numpy as np

from deepbullwhip._types import TimeSeries
from deepbullwhip.demand.base import DemandGenerator


class SemiconductorDemandGenerator(DemandGenerator):
    """AR(1) + seasonal + structural-shock demand, calibrated to WSTS data.

    Parameters
    ----------
    mu : float
        Mean monthly demand ($B/month), scaled internally to weekly.
    phi : float
        AR(1) autocorrelation coefficient.
    sigma_eps : float
        Residual coefficient of variation (sigma_eps / mu).
    seasonal_amp : float
        Seasonal amplitude as fraction of weekly mean.
    shock_period : int
        Period index when the structural shock begins.
    shock_magnitude : float
        Shock size as fraction of weekly mean.
    """

    def __init__(
        self,
        mu: float = 50.2,
        phi: float = 0.72,
        sigma_eps: float = 0.08,
        seasonal_amp: float = 0.06,
        shock_period: int = 104,
        shock_magnitude: float = 0.10,
    ) -> None:
        self.mu = mu
        self.phi = phi
        self.sigma_eps = sigma_eps
        self.seasonal_amp = seasonal_amp
        self.shock_period = shock_period
        self.shock_magnitude = shock_magnitude

    def generate(self, T: int = 156, seed: int | None = None) -> TimeSeries:
        rng = np.random.RandomState(seed)
        mu_weekly = self.mu / 4.33
        sigma_eps_abs = self.sigma_eps * mu_weekly

        D = np.zeros(T)
        D[0] = mu_weekly

        # Pre-compute deterministic components (vectorized)
        t_arr = np.arange(1, T)
        seasonal = self.seasonal_amp * mu_weekly * np.sin(2 * np.pi * t_arr / 52)
        shock = np.where(t_arr >= self.shock_period,
                         self.shock_magnitude * mu_weekly, 0.0)
        eps = rng.normal(0, sigma_eps_abs, T - 1)

        # AR(1) loop — sequential due to D[t-1] dependency
        for i, t in enumerate(t_arr):
            ar_term = mu_weekly + self.phi * (D[t - 1] - mu_weekly)
            D[t] = max(0.1, ar_term + seasonal[i] + shock[i] + eps[i])

        return D

    def generate_batch(
        self, T: int = 156, n_paths: int = 100, seed: int | None = None
    ) -> np.ndarray:
        """Generate N demand paths in parallel. Returns shape (n_paths, T).

        The AR(1) time dependency requires a sequential loop over T, but
        all N paths are updated simultaneously via vectorized operations.
        This is the primary GPU/vectorization opportunity: O(T) steps
        each processing N paths in parallel.

        Parameters
        ----------
        T : int
            Number of periods per path.
        n_paths : int
            Number of independent demand paths to generate.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        rng = np.random.RandomState(seed)
        mu_weekly = self.mu / 4.33
        sigma_eps_abs = self.sigma_eps * mu_weekly

        # Pre-compute all random innovations at once: (n_paths, T-1)
        eps = rng.normal(0, sigma_eps_abs, (n_paths, T - 1))

        # Pre-compute deterministic components: (T-1,)
        t_arr = np.arange(1, T)
        seasonal = self.seasonal_amp * mu_weekly * np.sin(2 * np.pi * t_arr / 52)
        shock = np.where(t_arr >= self.shock_period,
                         self.shock_magnitude * mu_weekly, 0.0)

        # Allocate output: (n_paths, T)
        D = np.zeros((n_paths, T))
        D[:, 0] = mu_weekly

        # Sequential over T (AR(1) dependency), vectorized over N
        for i in range(T - 1):
            ar_term = mu_weekly + self.phi * (D[:, i] - mu_weekly)
            D[:, i + 1] = np.maximum(
                0.1, ar_term + seasonal[i] + shock[i] + eps[:, i]
            )

        return D
