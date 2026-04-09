import numpy as np
import pytest

from deepbullwhip.sensitivity import compute_sensitivity


class _ConstantModel:
    def predict(self, X):
        return np.ones(X.shape[0]) * 5.0


class _LinearModel:
    """y = 2 * x[:, 0] + 1"""
    def predict(self, X):
        return 2 * X[:, 0] + 1


class TestComputeSensitivity:
    def test_constant_model_zero_sensitivity(self):
        model = _ConstantModel()
        X = np.random.RandomState(0).randn(50, 5)
        mean_s, std_s = compute_sensitivity(model, X)
        assert mean_s == pytest.approx(0.0, abs=1e-10)

    def test_linear_model_sensitivity(self):
        model = _LinearModel()
        X = np.random.RandomState(0).randn(50, 5) + 10  # ensure positive
        mean_s, _ = compute_sensitivity(model, X)
        # dy/dx0 = 2, so sensitivity should be ~2
        assert mean_s == pytest.approx(2.0, rel=0.01)

    def test_returns_tuple(self):
        model = _ConstantModel()
        X = np.random.RandomState(0).randn(10, 3)
        result = compute_sensitivity(model, X)
        assert isinstance(result, tuple)
        assert len(result) == 2
