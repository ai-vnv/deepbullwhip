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

        for t in range(1, T):
            ar_term = mu_weekly + self.phi * (D[t - 1] - mu_weekly)
            seasonal = self.seasonal_amp * mu_weekly * np.sin(2 * np.pi * t / 52)
            shock = self.shock_magnitude * mu_weekly if t >= self.shock_period else 0.0
            eps = rng.normal(0, sigma_eps_abs)
            D[t] = max(0.1, ar_term + seasonal + shock + eps)

        return D
