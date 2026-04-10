"""Tests for the new metrics module."""

import numpy as np
import pytest

from deepbullwhip import SemiconductorDemandGenerator, SerialSupplyChain
from deepbullwhip.metrics.bounds import ChenLowerBound
from deepbullwhip.metrics.bullwhip import BWR, CumulativeBWR
from deepbullwhip.metrics.cost import TotalCost
from deepbullwhip.metrics.inventory import FillRate, NSAmp
from deepbullwhip.registry import list_registered


@pytest.fixture
def simulation_result():
    gen = SemiconductorDemandGenerator()
    demand = gen.generate(T=156, seed=42)
    fm = np.full(156, np.mean(demand))
    fs = np.full(156, np.std(demand))
    chain = SerialSupplyChain()
    result = chain.simulate(demand, fm, fs)
    return result, demand


class TestBWR:
    def test_compute(self, simulation_result):
        result, demand = simulation_result
        bwr = BWR.compute(result, demand, echelon=0)
        assert bwr > 0
        # Should match the result's own bullwhip_ratio
        assert bwr == pytest.approx(result.echelon_results[0].bullwhip_ratio, rel=0.01)

    def test_registered(self):
        assert "BWR" in list_registered("metric")


class TestCumulativeBWR:
    def test_compute(self, simulation_result):
        result, demand = simulation_result
        cum_bwr = CumulativeBWR.compute(result, demand)
        assert cum_bwr > 0

    def test_registered(self):
        assert "CUM_BWR" in list_registered("metric")


class TestNSAmp:
    def test_compute(self, simulation_result):
        result, demand = simulation_result
        nsamp = NSAmp.compute(result, demand, echelon=0)
        assert nsamp > 0

    def test_registered(self):
        assert "NSAmp" in list_registered("metric")


class TestFillRate:
    def test_compute_bounded(self, simulation_result):
        result, demand = simulation_result
        fr = FillRate.compute(result, demand, echelon=0)
        assert 0 <= fr <= 1

    def test_registered(self):
        assert "FILL_RATE" in list_registered("metric")


class TestTotalCost:
    def test_compute_positive(self, simulation_result):
        result, demand = simulation_result
        tc = TotalCost.compute(result, demand, echelon=0)
        assert tc > 0

    def test_registered(self):
        assert "TC" in list_registered("metric")


class TestChenLowerBound:
    def test_compute_bound(self):
        bound = ChenLowerBound(lead_time=2, sensitivity=1.0, phi=0.72)
        val = bound.compute_bound()
        # 1 + 2*2*1*0.72/(1+0.72^2) + 4*1 = 1 + 2.88/1.5184 + 4 ≈ 6.897
        assert val == pytest.approx(6.897, abs=0.01)

    def test_registered(self):
        assert "ChenLowerBound" in list_registered("metric")


class TestBackwardCompat:
    """Ensure old diagnostics.metrics still works."""

    def test_old_bullwhip_ratio(self):
        from deepbullwhip.diagnostics.metrics import bullwhip_ratio

        orders = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
        demand = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        bw = bullwhip_ratio(orders, demand)
        assert bw > 0

    def test_old_fill_rate(self):
        from deepbullwhip.diagnostics.metrics import fill_rate

        inv = np.array([10.0, -5.0, 3.0, -1.0, 20.0])
        fr = fill_rate(inv)
        assert fr == pytest.approx(0.6)

    def test_old_cumulative_bullwhip(self):
        from deepbullwhip.diagnostics.metrics import cumulative_bullwhip

        ratios = [1.5, 2.0, 1.2]
        cum = cumulative_bullwhip(ratios)
        assert cum == pytest.approx(3.6)

    def test_old_lower_bound(self):
        from deepbullwhip.diagnostics.metrics import bullwhip_lower_bound

        bound = bullwhip_lower_bound(lead_time=2, sensitivity=1.0, phi=0.72)
        assert bound > 1.0
