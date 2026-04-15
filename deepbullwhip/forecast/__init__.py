"""Demand forecasting module."""

from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.forecast.exponential_smoothing import ExponentialSmoothingForecaster
from deepbullwhip.forecast.moving_average import MovingAverageForecaster
from deepbullwhip.forecast.naive import NaiveForecaster

try:
    from deepbullwhip.forecast.deepar import DeepARForecaster, DeepARTrainer  # noqa: F401
except ImportError:
    pass  # requires gluonts[torch] and torch

__all__ = [
    "Forecaster",
    "NaiveForecaster",
    "MovingAverageForecaster",
    "ExponentialSmoothingForecaster",
]
