import numpy as np
import pytest

from deepbullwhip.diagnostics.metrics import (
    bullwhip_lower_bound,
    bullwhip_ratio,
    cumulative_bullwhip,
    fill_rate,
)


class TestBullwhipRatio:
    def test_identical_signals(self):
        d = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
        assert bullwhip_ratio(d, d) == pytest.approx(1.0)

    def test_amplified_signal(self):
        d = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
        o = d * 2  # double variance
        assert bullwhip_ratio(o, d) == pytest.approx(4.0)

    def test_constant_demand(self):
        d = np.full(50, 10.0)
        o = np.random.RandomState(0).normal(10, 1, 50)
        # Constant demand has var=0, should return 1.0
        assert bullwhip_ratio(o, d) == 1.0


class TestFillRate:
    def test_all_positive(self):
        inv = np.array([1.0, 2.0, 0.5, 10.0])
        assert fill_rate(inv) == pytest.approx(1.0)

    def test_all_negative(self):
        inv = np.array([-1.0, -2.0, -0.5])
        assert fill_rate(inv) == pytest.approx(0.0)

    def test_mixed(self):
        inv = np.array([1.0, -1.0, 1.0, -1.0])
        assert fill_rate(inv) == pytest.approx(0.5)

    def test_zero_is_nonnegative(self):
        inv = np.array([0.0, 0.0])
        assert fill_rate(inv) == pytest.approx(1.0)


class TestCumulativeBullwhip:
    def test_single_echelon(self):
        assert cumulative_bullwhip([1.5]) == pytest.approx(1.5)

    def test_multiple_echelons(self):
        assert cumulative_bullwhip([1.2, 1.3, 1.1]) == pytest.approx(1.2 * 1.3 * 1.1)


class TestBullwhipLowerBound:
    def test_known_values(self):
        # L=2, lambda=0.5, phi=0.72
        # 1 + 2*2*0.5*0.72/(1+0.72^2) + 4*0.25
        L, lam, phi = 2, 0.5, 0.72
        expected = 1 + 2 * L * lam * phi / (1 + phi**2) + L**2 * lam**2
        assert bullwhip_lower_bound(L, lam, phi) == pytest.approx(expected)

    def test_zero_sensitivity(self):
        assert bullwhip_lower_bound(5, 0.0, 0.72) == pytest.approx(1.0)

    def test_increases_with_sensitivity(self):
        b1 = bullwhip_lower_bound(2, 0.1, 0.72)
        b2 = bullwhip_lower_bound(2, 0.5, 0.72)
        assert b2 > b1
