"""Single exponential smoothing forecaster."""

import numpy as np

from deepbullwhip._types import TimeSeries
from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.registry import register


@register("forecaster", "exponential_smoothing")
class ExponentialSmoothingForecaster(Forecaster):
    """Single exponential smoothing (SES) forecaster.

    Parameters
    ----------
    alpha : float
        Smoothing parameter in (0, 1]. Higher values weight recent
        observations more heavily. Default 0.3.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha

    def forecast(
        self, demand_history: TimeSeries, steps_ahead: int = 1
    ) -> tuple[float, float]:
        if len(demand_history) == 1:
            return float(demand_history[0]), 0.0

        # Compute SES level
        level = float(demand_history[0])
        errors = []
        for t in range(1, len(demand_history)):
            error = demand_history[t] - level
            errors.append(error)
            level = self.alpha * demand_history[t] + (1 - self.alpha) * level

        mean = level
        std = float(np.std(errors)) if len(errors) > 1 else 0.0
        return mean, std
