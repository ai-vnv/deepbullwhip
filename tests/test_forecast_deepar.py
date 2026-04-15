"""Tests for the DeepAR forecaster module."""

import numpy as np
import pytest

from deepbullwhip.forecast.deepar import DeepARForecaster, DeepARTrainer
from deepbullwhip.registry import list_registered

try:
    import gluonts  # noqa: F401
    _has_gluonts = True
except ImportError:
    _has_gluonts = False


class _MockSampleForecast:
    """Mimics gluonts SampleForecast with .samples array."""

    def __init__(self, samples: np.ndarray):
        self.samples = samples


class _MockPredictor:
    """Mimics a GluonTS predictor — returns deterministic sample forecasts."""

    def __init__(self, value: float = 10.0, std: float = 1.0):
        self.value = value
        self.std = std

    def predict(self, dataset, num_samples=100):
        for _ in dataset:
            samples = np.random.default_rng(0).normal(
                self.value, self.std, (num_samples, 1)
            )
            yield _MockSampleForecast(samples)


class _FailingPredictor:
    """Predictor that always raises."""

    def predict(self, dataset, num_samples=100):
        raise RuntimeError("inference failed")


# ── Registration ─────────────────────────────────────────────────────


class TestDeepARRegistration:
    def test_registered(self):
        assert "deepar" in list_registered("forecaster")


# ── DeepARForecaster without predictor (fallback paths) ─────────────


class TestDeepARFallback:
    def test_no_predictor_returns_naive(self):
        fc = DeepARForecaster(predictor=None)
        demand = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
        mean, std = fc.forecast(demand)
        assert mean == pytest.approx(np.mean(demand))
        assert std == pytest.approx(np.std(demand))

    def test_short_history_single_element(self):
        fc = DeepARForecaster(predictor=None)
        mean, std = fc.forecast(np.array([5.0]))
        assert mean == pytest.approx(5.0)
        assert std == pytest.approx(0.0)

    def test_short_history_two_elements(self):
        fc = DeepARForecaster(predictor=None)
        mean, std = fc.forecast(np.array([4.0, 6.0]))
        assert mean == pytest.approx(5.0)
        assert std == pytest.approx(1.0)

    def test_generate_forecasts_no_predictor(self):
        fc = DeepARForecaster(predictor=None)
        demand = np.random.default_rng(42).normal(10, 2, 50)
        fm, fs = fc.generate_forecasts(demand)
        assert fm.shape == (50,)
        assert fs.shape == (50,)
        # First element should be the mean of just demand[0]
        assert fm[0] == pytest.approx(demand[0])


# ── DeepARForecaster with mock predictor ────────────────────────────


@pytest.mark.skipif(not _has_gluonts, reason="gluonts[torch] not installed")
class TestDeepARWithPredictor:
    def test_forecast_returns_mean_std(self):
        mock = _MockPredictor(value=10.0, std=1.0)
        fc = DeepARForecaster(predictor=mock, context_length=5, num_samples=50)
        demand = np.full(20, 10.0)
        mean, std = fc.forecast(demand)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert mean == pytest.approx(10.0, abs=1.0)
        assert std > 0

    def test_forecast_respects_context_length(self):
        mock = _MockPredictor(value=10.0, std=0.5)
        fc = DeepARForecaster(predictor=mock, context_length=10, num_samples=50)
        # Even with long history, should work fine
        demand = np.full(100, 10.0)
        mean, std = fc.forecast(demand)
        assert isinstance(mean, float)

    def test_forecast_fallback_on_exception(self):
        fc = DeepARForecaster(
            predictor=_FailingPredictor(), context_length=5, num_samples=50
        )
        demand = np.array([10.0, 12.0, 8.0, 11.0, 9.0, 10.5])
        mean, std = fc.forecast(demand)
        # Should fall back to naive stats over last context_length elements
        expected_mean = float(np.mean(demand[-5:]))
        expected_std = float(np.std(demand[-5:]))
        assert mean == pytest.approx(expected_mean)
        assert std == pytest.approx(expected_std)

    def test_forecast_short_history_with_predictor(self):
        """With predictor but <3 elements, should still fallback."""
        mock = _MockPredictor(value=10.0)
        fc = DeepARForecaster(predictor=mock, context_length=5)
        mean, std = fc.forecast(np.array([5.0, 7.0]))
        assert mean == pytest.approx(6.0)

    def test_generate_forecasts_with_predictor(self):
        mock = _MockPredictor(value=10.0, std=0.5)
        fc = DeepARForecaster(predictor=mock, context_length=5, num_samples=50)
        demand = np.full(20, 10.0)
        fm, fs = fc.generate_forecasts(demand)
        assert fm.shape == (20,)
        assert fs.shape == (20,)
        # After warmup, values should come from mock predictor
        assert fm[-1] == pytest.approx(10.0, abs=1.0)

    def test_generate_forecasts_short_demand(self):
        """Demand shorter than context_length — all warmup, no batch."""
        mock = _MockPredictor(value=10.0, std=0.5)
        fc = DeepARForecaster(predictor=mock, context_length=100, num_samples=50)
        demand = np.full(10, 10.0)
        fm, fs = fc.generate_forecasts(demand)
        assert fm.shape == (10,)
        assert fs.shape == (10,)

    def test_generate_forecasts_fallback_on_exception(self):
        fc = DeepARForecaster(
            predictor=_FailingPredictor(), context_length=5, num_samples=50
        )
        demand = np.random.default_rng(42).normal(10, 2, 30)
        fm, fs = fc.generate_forecasts(demand)
        assert fm.shape == (30,)
        assert fs.shape == (30,)


# ── DeepARForecaster init params ────────────────────────────────────


class TestDeepARInit:
    def test_default_params(self):
        fc = DeepARForecaster()
        assert fc._predictor is None
        assert fc.context_length == 52
        assert fc.num_samples == 200
        assert fc.freq == "W"

    def test_custom_params(self):
        fc = DeepARForecaster(
            context_length=26, num_samples=100, freq="M"
        )
        assert fc.context_length == 26
        assert fc.num_samples == 100
        assert fc.freq == "M"


# ── DeepARTrainer init ──────────────────────────────────────────────


class TestDeepARTrainer:
    def test_default_params(self):
        t = DeepARTrainer()
        assert t.freq == "W"
        assert t.prediction_length == 1
        assert t.context_length == 52
        assert t.epochs == 50
        assert t.num_layers == 3
        assert t.hidden_size == 40
        assert t.lr == 1e-3
        assert t.batch_size == 64
        assert t.num_samples == 200
        assert t.likelihood == "gaussian"

    def test_custom_params(self):
        t = DeepARTrainer(
            freq="M",
            prediction_length=3,
            context_length=24,
            epochs=10,
            num_layers=2,
            hidden_size=20,
            lr=5e-4,
            batch_size=32,
            num_samples=100,
            likelihood="negative_binomial",
        )
        assert t.freq == "M"
        assert t.prediction_length == 3
        assert t.context_length == 24
        assert t.epochs == 10
        assert t.num_layers == 2
        assert t.hidden_size == 20
        assert t.batch_size == 32

    @pytest.mark.skipif(
        not _has_gluonts, reason="gluonts[torch] not installed"
    )
    def test_train_returns_forecaster(self):
        """Train with minimal config (1 epoch, tiny model) to verify pipeline."""
        trainer = DeepARTrainer(
            freq="W",
            prediction_length=1,
            context_length=10,
            epochs=1,
            num_layers=1,
            hidden_size=10,
            batch_size=16,
            num_samples=10,
        )
        rng = np.random.default_rng(42)
        series = [rng.normal(10, 2, 60) for _ in range(5)]
        fc = trainer.train(series)
        assert isinstance(fc, DeepARForecaster)
        assert fc._predictor is not None
        assert fc.context_length == 10
        assert fc.num_samples == 10

        # Verify the trained forecaster can produce forecasts
        demand = rng.normal(10, 2, 30)
        mean, std = fc.forecast(demand)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert np.isfinite(mean)
        assert np.isfinite(std)

    @pytest.mark.skipif(
        not _has_gluonts, reason="gluonts[torch] not installed"
    )
    def test_trained_generate_forecasts(self):
        """Verify batch rolling forecast works with a trained model."""
        trainer = DeepARTrainer(
            freq="W",
            prediction_length=1,
            context_length=10,
            epochs=1,
            num_layers=1,
            hidden_size=10,
            batch_size=16,
            num_samples=10,
        )
        rng = np.random.default_rng(99)
        series = [rng.normal(10, 2, 60) for _ in range(5)]
        fc = trainer.train(series)

        demand = rng.normal(10, 2, 30)
        fm, fs = fc.generate_forecasts(demand)
        assert fm.shape == (30,)
        assert fs.shape == (30,)
        assert np.all(np.isfinite(fm))
        assert np.all(np.isfinite(fs))
