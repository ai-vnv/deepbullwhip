"""Tests for new demand generators: BeerGame, ARMA, Replay."""

import numpy as np
import pytest

from deepbullwhip.demand.arma import ARMADemandGenerator
from deepbullwhip.demand.beer_game import BeerGameDemandGenerator
from deepbullwhip.demand.replay import ReplayDemandGenerator
from deepbullwhip.registry import list_registered


class TestBeerGameDemand:
    def test_step_at_correct_time(self):
        gen = BeerGameDemandGenerator(step_time=5)
        d = gen.generate(T=20)
        assert d[4] == 4.0
        assert d[5] == 8.0

    def test_length(self):
        gen = BeerGameDemandGenerator()
        d = gen.generate(T=100)
        assert len(d) == 100

    def test_batch_deterministic(self):
        gen = BeerGameDemandGenerator()
        batch = gen.generate_batch(T=52, n_paths=10)
        assert batch.shape == (10, 52)
        # All paths should be identical (deterministic)
        np.testing.assert_array_equal(batch[0], batch[5])

    def test_custom_params(self):
        gen = BeerGameDemandGenerator(base_demand=2.0, step_demand=10.0, step_time=3)
        d = gen.generate(T=10)
        assert d[2] == 2.0
        assert d[3] == 10.0

    def test_registered(self):
        assert "beer_game" in list_registered("demand")


class TestARMADemand:
    def test_ar1_shape(self):
        gen = ARMADemandGenerator(ar_coeffs=[0.7], ma_coeffs=[])
        d = gen.generate(T=100, seed=42)
        assert len(d) == 100

    def test_positivity(self):
        gen = ARMADemandGenerator(mu=12.5, sigma=1.0)
        d = gen.generate(T=200, seed=42)
        assert np.all(d > 0)

    def test_reproducibility(self):
        gen = ARMADemandGenerator()
        d1 = gen.generate(T=100, seed=42)
        d2 = gen.generate(T=100, seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_arma_with_ma(self):
        gen = ARMADemandGenerator(ar_coeffs=[0.5], ma_coeffs=[0.3], mu=10.0)
        d = gen.generate(T=100, seed=42)
        assert len(d) == 100
        assert np.all(d > 0)

    def test_batch_shape(self):
        gen = ARMADemandGenerator()
        batch = gen.generate_batch(T=50, n_paths=20, seed=42)
        assert batch.shape == (20, 50)

    def test_registered(self):
        assert "arma" in list_registered("demand")


class TestReplayDemand:
    def test_exact_length(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gen = ReplayDemandGenerator(data=data)
        d = gen.generate(T=5)
        np.testing.assert_array_equal(d, data)

    def test_cycling(self):
        data = np.array([1.0, 2.0, 3.0])
        gen = ReplayDemandGenerator(data=data)
        d = gen.generate(T=7)
        expected = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])
        np.testing.assert_array_equal(d, expected)

    def test_batch_paths_differ(self):
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        gen = ReplayDemandGenerator(data=data)
        batch = gen.generate_batch(T=5, n_paths=10, seed=42)
        assert batch.shape == (10, 5)
        # Paths should differ due to noise
        assert not np.allclose(batch[0], batch[1])

    def test_batch_non_negative(self):
        data = np.array([10.0, 20.0, 30.0])
        gen = ReplayDemandGenerator(data=data)
        batch = gen.generate_batch(T=10, n_paths=50, seed=42)
        assert np.all(batch >= 0)

    def test_from_list(self):
        gen = ReplayDemandGenerator(data=[5.0, 10.0, 15.0])
        d = gen.generate(T=3)
        np.testing.assert_array_equal(d, [5.0, 10.0, 15.0])

    def test_registered(self):
        assert "replay" in list_registered("demand")
