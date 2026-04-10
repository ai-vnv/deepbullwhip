"""Synthetic demand datasets (convenience wrappers)."""

from __future__ import annotations

from deepbullwhip._types import TimeSeries


def load_ar1(
    T: int = 156,
    mu: float = 12.5,
    rho: float = 0.7,
    sigma: float = 1.0,
    seed: int = 42,
) -> TimeSeries:
    """Generate an AR(1) demand series.

    Parameters
    ----------
    T : int
        Number of periods.
    mu : float
        Long-run mean.
    rho : float
        AR(1) coefficient.
    sigma : float
        Innovation std dev.
    seed : int
        Random seed.

    Returns
    -------
    TimeSeries, shape (T,)
    """
    from deepbullwhip.demand.arma import ARMADemandGenerator

    gen = ARMADemandGenerator(ar_coeffs=[rho], ma_coeffs=[], mu=mu, sigma=sigma)
    return gen.generate(T=T, seed=seed)


def load_arma(
    T: int = 156,
    ar_coeffs: list[float] | None = None,
    ma_coeffs: list[float] | None = None,
    mu: float = 12.5,
    sigma: float = 1.0,
    seed: int = 42,
) -> TimeSeries:
    """Generate an ARMA(p,q) demand series.

    Parameters
    ----------
    T : int
        Number of periods.
    ar_coeffs : list[float]
        AR coefficients.
    ma_coeffs : list[float]
        MA coefficients.
    mu : float
        Long-run mean.
    sigma : float
        Innovation std dev.
    seed : int
        Random seed.

    Returns
    -------
    TimeSeries, shape (T,)
    """
    from deepbullwhip.demand.arma import ARMADemandGenerator

    gen = ARMADemandGenerator(
        ar_coeffs=ar_coeffs, ma_coeffs=ma_coeffs, mu=mu, sigma=sigma
    )
    return gen.generate(T=T, seed=seed)
