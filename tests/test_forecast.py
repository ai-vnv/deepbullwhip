"""Tests for the forecaster module."""

import numpy as np
import pytest

from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.forecast.exponential_smoothing import ExponentialSmoothingForecaster
from deepbullwhip.forecast.moving_average import MovingAverageForecaster
from deepbullwhip.forecast.naive import NaiveForecaster
from deepbullwhip.registry import list_registered


class TestNaiveForecaster:
    def test_constant_series(self):
        f = NaiveForecaster()
        demand = np.full(50, 10.0)
        fm, fs = f.forecast(demand)
        assert fm == pytest.approx(10.0)
        assert fs == pytest.approx(0.0)

    def test_generate_forecasts_shape(self):
        f = NaiveForecaster()
        demand = np.random.default_rng(42).normal(10, 2, 100)
        fm, fs = f.generate_forecasts(demand)
        assert fm.shape == (100,)
        assert fs.shape == (100,)

    def test_registered(self):
        assert "naive" in list_registered("forecaster")


class TestMovingAverageForecaster:
    def test_window_mean(self):
        f = MovingAverageForecaster(window=3)
        demand = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        fm, fs = f.forecast(demand)
        # Last 3: [10, 20, 30]
        assert fm == pytest.approx(20.0)

    def test_short_history(self):
        f = MovingAverageForecaster(window=10)
        demand = np.array([5.0, 10.0])
        fm, fs = f.forecast(demand)
        assert fm == pytest.approx(7.5)

    def test_generate_forecasts_shape(self):
        f = MovingAverageForecaster(window=5)
        demand = np.random.default_rng(42).normal(10, 2, 100)
        fm, fs = f.generate_forecasts(demand)
        assert fm.shape == (100,)
        assert fs.shape == (100,)

    def test_registered(self):
        assert "moving_average" in list_registered("forecaster")


class TestExponentialSmoothingForecaster:
    def test_convergence_to_mean(self):
        """SES on a constant series should converge to the constant."""
        f = ExponentialSmoothingForecaster(alpha=0.3)
        demand = np.full(100, 10.0)
        fm, fs = f.forecast(demand)
        assert fm == pytest.approx(10.0, abs=0.01)

    def test_single_observation(self):
        f = ExponentialSmoothingForecaster(alpha=0.5)
        fm, fs = f.forecast(np.array([7.0]))
        assert fm == pytest.approx(7.0)
        assert fs == pytest.approx(0.0)

    def test_generate_forecasts_shape(self):
        f = ExponentialSmoothingForecaster(alpha=0.3)
        demand = np.random.default_rng(42).normal(10, 2, 100)
        fm, fs = f.generate_forecasts(demand)
        assert fm.shape == (100,)
        assert fs.shape == (100,)

    def test_registered(self):
        assert "exponential_smoothing" in list_registered("forecaster")


class TestForecasterABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Forecaster()

    def test_all_registered(self):
        registered = list_registered("forecaster")
        assert "naive" in registered
        assert "moving_average" in registered
        assert "exponential_smoothing" in registered
