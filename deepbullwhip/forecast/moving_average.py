"""Moving average forecaster."""

import numpy as np

from deepbullwhip._types import TimeSeries
from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.registry import register


@register("forecaster", "moving_average")
class MovingAverageForecaster(Forecaster):
    """Rolling window moving average forecast.

    Chen et al. (2000) derived bullwhip ratio bounds for the
    moving average forecasting method with OUT policy.

    Parameters
    ----------
    window : int
        Number of recent periods to average (default 10).
    """

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def forecast(
        self, demand_history: TimeSeries, steps_ahead: int = 1
    ) -> tuple[float, float]:
        recent = demand_history[-self.window :]
        mean = float(np.mean(recent))
        std = float(np.std(recent)) if len(recent) > 1 else 0.0
        return mean, std
