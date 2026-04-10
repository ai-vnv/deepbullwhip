"""Naive (constant) forecaster using sample statistics."""

import numpy as np

from deepbullwhip._types import TimeSeries
from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.registry import register


@register("forecaster", "naive")
class NaiveForecaster(Forecaster):
    """Constant forecast using sample mean and std of observed history.

    This replicates the default forecasting approach used in the current
    ``simulate()`` interface where fm and fs are precomputed constants.
    """

    def forecast(
        self, demand_history: TimeSeries, steps_ahead: int = 1
    ) -> tuple[float, float]:
        mean = float(np.mean(demand_history))
        std = float(np.std(demand_history)) if len(demand_history) > 1 else 0.0
        return mean, std
