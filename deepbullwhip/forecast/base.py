"""Abstract base class for demand forecasters."""

from abc import ABC, abstractmethod

import numpy as np

from deepbullwhip._types import TimeSeries


class Forecaster(ABC):
    """Abstract base class for demand forecasters.

    A Forecaster produces (mean, std) estimates for each period given
    the demand history up to that point. Used by BenchmarkRunner to
    generate the ``forecasts_mean`` / ``forecasts_std`` arrays that
    ``simulate()`` requires.
    """

    @abstractmethod
    def forecast(
        self, demand_history: TimeSeries, steps_ahead: int = 1
    ) -> tuple[float, float]:
        """Return (forecast_mean, forecast_std) given history.

        Parameters
        ----------
        demand_history : TimeSeries
            Observed demand up to (and including) current period.
        steps_ahead : int
            Forecast horizon (default 1).

        Returns
        -------
        tuple[float, float]
            (point forecast, forecast std dev)
        """
        ...

    def generate_forecasts(
        self, demand: TimeSeries
    ) -> tuple[TimeSeries, TimeSeries]:
        """Run rolling forecast over a full demand series.

        Parameters
        ----------
        demand : TimeSeries, shape (T,)

        Returns
        -------
        fm : TimeSeries, shape (T,) forecast means
        fs : TimeSeries, shape (T,) forecast stds
        """
        T = len(demand)
        fm = np.zeros(T)
        fs = np.zeros(T)
        for t in range(T):
            history = demand[: t + 1]
            fm[t], fs[t] = self.forecast(history)
        return fm, fs
