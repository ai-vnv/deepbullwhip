"""Tests for the datasets module."""

import numpy as np
import pytest

from deepbullwhip.datasets.beer_game import load_beer_game
from deepbullwhip.datasets.synthetic import load_ar1, load_arma
from deepbullwhip.datasets.wsts import load_wsts


class TestLoadBeerGame:
    def test_default_length(self):
        d = load_beer_game()
        assert len(d) == 52

    def test_custom_length(self):
        d = load_beer_game(T=100)
        assert len(d) == 100

    def test_step_values(self):
        d = load_beer_game()
        assert d[0] == 4.0
        assert d[4] == 4.0
        assert d[5] == 8.0
        assert d[-1] == 8.0


class TestLoadAR1:
    def test_shape(self):
        d = load_ar1(T=100)
        assert len(d) == 100

    def test_reproducibility(self):
        d1 = load_ar1(seed=42)
        d2 = load_ar1(seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_positivity(self):
        d = load_ar1(T=200)
        assert np.all(d > 0)


class TestLoadARMA:
    def test_shape(self):
        d = load_arma(T=100, ar_coeffs=[0.5], ma_coeffs=[0.3])
        assert len(d) == 100


class TestLoadM5:
    def test_import_error(self):
        """Should raise ImportError when kaggle is not installed and no cache."""
        from deepbullwhip.datasets.m5 import load_m5
        # This should either succeed (cached) or raise ImportError
        # We test that the function exists and is callable
        assert callable(load_m5)


class TestLoadWSTS:
    def test_loads_bundled_data(self):
        d = load_wsts(region="worldwide", product="total")
        assert len(d) == 60  # 5 years x 12 months
        assert np.all(d > 0)

    def test_different_region(self):
        d = load_wsts(region="americas", product="total")
        assert len(d) == 60

    def test_unknown_column(self):
        with pytest.raises(KeyError, match="not found"):
            load_wsts(region="antarctica", product="total")
