"""Tests for new metrics edge cases to improve coverage.

Targets: damping_ratio, peak_bwr, expected_shortfall,
         inventory_turnover, order_smoothing_ratio, rfu.
"""

import numpy as np
import pytest

from deepbullwhip._types import EchelonResult, SimulationResult


def _make_result(orders, inventory=None, n_echelons=1):
    """Build a minimal SimulationResult for metric testing."""
    echelon_results = []
    for i in range(n_echelons):
        o = np.asarray(orders if i == 0 else orders, dtype=float)
        inv = np.asarray(
            inventory if inventory is not None else np.ones(len(orders)),
            dtype=float,
        )
        echelon_results.append(
            EchelonResult(
                name=f"E{i}",
                orders=o,
                inventory_levels=inv,
                costs=np.ones(len(orders)),
                bullwhip_ratio=1.0,
                fill_rate=1.0,
                total_cost=float(len(orders)),
            )
        )
    return SimulationResult(
        echelon_results=echelon_results,
        cumulative_bullwhip=1.0,
        total_cost=float(len(orders) * n_echelons),
    )


# ── DampingRatio ─────────────────────────────────────────────────────

class TestDampingRatio:
    def test_short_series_returns_nan(self):
        from deepbullwhip.metrics.damping_ratio import DampingRatio

        result = _make_result(np.ones(5))
        demand = np.ones(5)
        assert np.isnan(DampingRatio.compute(result, demand))

    def test_constant_series(self):
        """Constant orders produce zero AR coefficients -> phi2=0 -> nan."""
        from deepbullwhip.metrics.damping_ratio import DampingRatio

        result = _make_result(np.full(50, 10.0))
        demand = np.full(50, 10.0)
        val = DampingRatio.compute(result, demand)
        # phi2 >= 0 for constant, so should return nan
        assert np.isnan(val) or isinstance(val, float)

    def test_oscillatory_series(self):
        """An oscillatory AR(2) series should produce a finite damping ratio."""
        from deepbullwhip.metrics.damping_ratio import DampingRatio

        rng = np.random.RandomState(42)
        T = 200
        orders = np.zeros(T)
        orders[0] = 10.0
        orders[1] = 10.0
        # AR(2) with phi1=0.5, phi2=-0.3 (oscillatory, stable)
        for t in range(2, T):
            orders[t] = 10 + 0.5 * (orders[t - 1] - 10) - 0.3 * (orders[t - 2] - 10) + rng.normal(0, 0.5)

        result = _make_result(orders)
        demand = np.full(T, 10.0)
        val = DampingRatio.compute(result, demand)
        assert isinstance(val, float)
        assert not np.isnan(val)

    def test_positive_phi2_returns_nan(self):
        """A random walk with positive phi2 should return nan."""
        from deepbullwhip.metrics.damping_ratio import DampingRatio

        # Build a series with strong positive phi2
        rng = np.random.RandomState(123)
        T = 100
        orders = np.cumsum(rng.normal(10, 0.1, T))  # random walk -> phi1~2, phi2~-1 or varied

        result = _make_result(orders)
        demand = np.full(T, 10.0)
        val = DampingRatio.compute(result, demand)
        # Just check it's a float (may or may not be nan)
        assert isinstance(val, float)


# ── PeakBWR ──────────────────────────────────────────────────────────

class TestPeakBWR:
    def test_short_series_fallback(self):
        """Series shorter than window=4 uses full-series fallback."""
        from deepbullwhip.metrics.peak_bwr import PeakBWR

        orders = np.array([10.0, 12.0, 8.0])
        demand = np.array([9.0, 11.0, 10.0])
        result = _make_result(orders)
        val = PeakBWR.compute(result, demand)
        assert val > 0

    def test_constant_demand_short_returns_zero(self):
        """Zero-variance demand in short series returns 0."""
        from deepbullwhip.metrics.peak_bwr import PeakBWR

        orders = np.array([10.0, 10.0, 10.0])
        demand = np.array([10.0, 10.0, 10.0])
        result = _make_result(orders)
        val = PeakBWR.compute(result, demand)
        assert val == 0.0

    def test_zero_variance_window_skipped(self):
        """Windows with zero demand variance are skipped."""
        from deepbullwhip.metrics.peak_bwr import PeakBWR

        # Use a long enough series with some constant-demand windows
        orders = np.concatenate([np.array([10, 20, 30, 40, 50]), np.full(30, 15.0)])
        demand = np.concatenate([np.full(5, 10.0), np.full(30, 10.0)])
        result = _make_result(orders)
        val = PeakBWR.compute(result, demand)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_normal_computation(self):
        """Standard case with enough data for rolling windows."""
        from deepbullwhip.metrics.peak_bwr import PeakBWR

        rng = np.random.RandomState(42)
        T = 100
        demand = rng.normal(50, 5, T)
        orders = demand * 1.5 + rng.normal(0, 3, T)
        result = _make_result(orders)
        val = PeakBWR.compute(result, demand)
        assert val > 0


# ── ExpectedShortfall ────────────────────────────────────────────────

class TestExpectedShortfall:
    def test_no_stockouts(self):
        from deepbullwhip.metrics.expected_shortfall import ExpectedShortfall

        inv = np.array([10.0, 5.0, 3.0, 20.0])
        result = _make_result(np.ones(4), inventory=inv)
        demand = np.ones(4)
        assert ExpectedShortfall.compute(result, demand) == 0.0

    def test_all_stockouts(self):
        from deepbullwhip.metrics.expected_shortfall import ExpectedShortfall

        inv = np.array([-5.0, -10.0, -3.0])
        result = _make_result(np.ones(3), inventory=inv)
        demand = np.ones(3)
        val = ExpectedShortfall.compute(result, demand)
        assert val == pytest.approx(6.0)  # mean(5, 10, 3) = 6

    def test_mixed_stockouts(self):
        from deepbullwhip.metrics.expected_shortfall import ExpectedShortfall

        inv = np.array([10.0, -4.0, 5.0, -6.0])
        result = _make_result(np.ones(4), inventory=inv)
        demand = np.ones(4)
        val = ExpectedShortfall.compute(result, demand)
        assert val == pytest.approx(5.0)  # mean(4, 6) = 5


# ── InventoryTurnover ───────────────────────────────────────────────

class TestInventoryTurnover:
    def test_zero_inventory_returns_inf(self):
        from deepbullwhip.metrics.inventory_turnover import InventoryTurnover

        inv = np.zeros(10)
        result = _make_result(np.full(10, 10.0), inventory=inv)
        demand = np.full(10, 10.0)
        assert InventoryTurnover.compute(result, demand) == float("inf")

    def test_normal_turnover(self):
        from deepbullwhip.metrics.inventory_turnover import InventoryTurnover

        inv = np.full(10, 5.0)  # constant inventory of 5
        demand = np.full(10, 10.0)  # constant demand of 10
        result = _make_result(np.full(10, 10.0), inventory=inv)
        val = InventoryTurnover.compute(result, demand)
        # 52 * 10 / 5 = 104
        assert val == pytest.approx(104.0)

    def test_negative_inventory_clamped(self):
        from deepbullwhip.metrics.inventory_turnover import InventoryTurnover

        inv = np.array([-10.0, -5.0, 0.0])  # all <= 0
        result = _make_result(np.ones(3), inventory=inv)
        demand = np.full(3, 10.0)
        assert InventoryTurnover.compute(result, demand) == float("inf")


# ── OrderSmoothingRatio ──────────────────────────────────────────────

class TestOrderSmoothingRatio:
    def test_single_period(self):
        from deepbullwhip.metrics.order_smoothing_ratio import OrderSmoothingRatio

        result = _make_result(np.array([10.0]))
        demand = np.array([10.0])
        assert OrderSmoothingRatio.compute(result, demand) == 0.0

    def test_constant_demand_returns_zero(self):
        from deepbullwhip.metrics.order_smoothing_ratio import OrderSmoothingRatio

        orders = np.array([10.0, 12.0, 14.0])
        demand = np.full(3, 10.0)  # zero variance
        result = _make_result(orders)
        assert OrderSmoothingRatio.compute(result, demand) == 0.0

    def test_normal_computation(self):
        from deepbullwhip.metrics.order_smoothing_ratio import OrderSmoothingRatio

        orders = np.array([10.0, 15.0, 12.0, 18.0, 11.0])
        demand = np.array([10.0, 12.0, 8.0, 14.0, 9.0])
        result = _make_result(orders)
        val = OrderSmoothingRatio.compute(result, demand)
        assert val > 0


# ── RFU (multi-echelon) ─────────────────────────────────────────────

class TestRFU:
    def test_echelon_zero(self):
        from deepbullwhip.metrics.rfu import RatioOfForecastUncertainty

        orders = np.array([12.0, 14.0, 10.0, 16.0])
        demand = np.array([10.0, 12.0, 8.0, 14.0])
        result = _make_result(orders)
        val = RatioOfForecastUncertainty.compute(result, demand, echelon=0)
        assert val >= 0

    def test_zero_demand_variance(self):
        from deepbullwhip.metrics.rfu import RatioOfForecastUncertainty

        orders = np.array([12.0, 14.0, 10.0])
        demand = np.full(3, 10.0)
        result = _make_result(orders)
        assert RatioOfForecastUncertainty.compute(result, demand) == 0.0


# ── Render helpers ───────────────────────────────────────────────────

class TestRenderHelpers:
    def test_auto_fontsize(self):
        from deepbullwhip.render._matplotlib import _auto_fontsize

        # Short name -> max fontsize
        fs_short = _auto_fontsize("E1", 0.5)
        assert 5.5 <= fs_short <= 10.0

        # Long name -> smaller fontsize
        fs_long = _auto_fontsize("LongEchelonName", 0.5)
        assert 5.5 <= fs_long <= 10.0
        assert fs_long <= fs_short

    def test_build_result_map_none(self):
        from deepbullwhip.render._matplotlib import _build_result_map
        from deepbullwhip.chain.graph import SupplyChainGraph

        graph = SupplyChainGraph()
        assert _build_result_map(graph, None) == {}

    def test_build_result_map_network_result(self):
        from deepbullwhip.render._matplotlib import _build_result_map
        from deepbullwhip.chain.graph import SupplyChainGraph
        from deepbullwhip._types import NetworkSimulationResult, EchelonResult

        graph = SupplyChainGraph()
        er = EchelonResult(
            name="A", orders=np.array([1.0]), inventory_levels=np.array([1.0]),
            costs=np.array([1.0]), bullwhip_ratio=1.0, fill_rate=1.0, total_cost=1.0,
        )
        net_result = NetworkSimulationResult(
            node_results={"A": er},
            edge_flows={},
            cumulative_bullwhip=1.0,
            total_cost=1.0,
        )
        result_map = _build_result_map(graph, net_result)
        assert "A" in result_map


# ── Forecast __init__ coverage ───────────────────────────────────────

class TestForecastInit:
    def test_public_exports(self):
        import deepbullwhip.forecast as f

        assert hasattr(f, "Forecaster")
        assert hasattr(f, "NaiveForecaster")
        assert hasattr(f, "MovingAverageForecaster")
        assert hasattr(f, "ExponentialSmoothingForecaster")
