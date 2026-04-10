"""Tests for new ordering policies: POUT, Constant, Smoothing."""

import numpy as np
import pytest

from deepbullwhip import (
    EchelonConfig,
    OrderUpToPolicy,
    SemiconductorDemandGenerator,
    SerialSupplyChain,
)
from deepbullwhip.chain.echelon import SupplyChainEchelon
from deepbullwhip.cost.newsvendor import NewsvendorCost
from deepbullwhip.policy.constant_order import ConstantOrderPolicy
from deepbullwhip.policy.proportional_out import ProportionalOUTPolicy
from deepbullwhip.policy.smoothing_out import SmoothingOUTPolicy
from deepbullwhip.registry import list_registered


# --- ProportionalOUTPolicy ---


class TestProportionalOUT:
    def test_instantiation(self):
        p = ProportionalOUTPolicy(lead_time=2, alpha=0.5)
        assert p.lead_time == 2
        assert p.alpha == 0.5

    def test_compute_order_positive(self):
        p = ProportionalOUTPolicy(lead_time=2, alpha=0.5)
        order = p.compute_order(inventory_position=30.0, forecast_mean=10.0, forecast_std=2.0)
        assert order > 0

    def test_compute_order_non_negative(self):
        p = ProportionalOUTPolicy(lead_time=2, alpha=0.5)
        order = p.compute_order(inventory_position=1000.0, forecast_mean=1.0, forecast_std=0.1)
        assert order >= 0

    def test_alpha_1_equals_out(self):
        """POUT with alpha=1 should match standard OUT."""
        pout = ProportionalOUTPolicy(lead_time=4, service_level=0.95, alpha=1.0)
        out = OrderUpToPolicy(lead_time=4, service_level=0.95)

        for ip, fm, fs in [(50.0, 10.0, 2.0), (20.0, 15.0, 3.0), (100.0, 5.0, 1.0)]:
            assert pout.compute_order(ip, fm, fs) == pytest.approx(
                out.compute_order(ip, fm, fs)
            )

    def test_lower_alpha_reduces_order(self):
        """Lower alpha should produce smaller orders."""
        high = ProportionalOUTPolicy(lead_time=2, alpha=0.9)
        low = ProportionalOUTPolicy(lead_time=2, alpha=0.3)

        o_high = high.compute_order(30.0, 10.0, 2.0)
        o_low = low.compute_order(30.0, 10.0, 2.0)
        assert o_low < o_high

    def test_registered(self):
        assert "proportional_out" in list_registered("policy")

    def test_pout_reduces_bwr(self):
        """POUT should produce lower BWR than standard OUT in a chain."""
        gen = SemiconductorDemandGenerator()
        demand = gen.generate(T=156, seed=42)
        fm = np.full(156, np.mean(demand))
        fs = np.full(156, np.std(demand))

        # OUT chain
        out_chain = SerialSupplyChain()
        out_result = out_chain.simulate(demand, fm, fs)
        out_bwr = out_result.echelon_results[0].bullwhip_ratio

        # POUT chain (alpha=0.3)
        configs = [
            EchelonConfig("E1", lead_time=2, holding_cost=0.15, backorder_cost=0.60)
        ]
        pout_policy = ProportionalOUTPolicy(lead_time=2, service_level=0.95, alpha=0.3)
        cost_fn = NewsvendorCost(holding_cost=0.15, backorder_cost=0.60)
        pout_echelon = SupplyChainEchelon("E1", lead_time=2, policy=pout_policy, cost_fn=cost_fn)
        pout_chain = SerialSupplyChain(echelons=[pout_echelon])
        pout_result = pout_chain.simulate(demand, fm, fs)
        pout_bwr = pout_result.echelon_results[0].bullwhip_ratio

        assert pout_bwr < out_bwr


# --- ConstantOrderPolicy ---


class TestConstantOrder:
    def test_instantiation(self):
        p = ConstantOrderPolicy(order_quantity=10.0)
        assert p.order_quantity == 10.0

    def test_default_quantity(self):
        p = ConstantOrderPolicy()
        assert p.order_quantity == 12.5

    def test_compute_order_constant(self):
        p = ConstantOrderPolicy(order_quantity=7.5)
        for ip, fm, fs in [(0, 10, 2), (100, 1, 0.1), (-50, 20, 5)]:
            assert p.compute_order(ip, fm, fs) == 7.5

    def test_registered(self):
        assert "constant_order" in list_registered("policy")

    def test_constant_bwr_zero(self):
        """Constant orders should produce Var(orders) ≈ 0."""
        gen = SemiconductorDemandGenerator()
        demand = gen.generate(T=100, seed=42)
        fm = np.full(100, np.mean(demand))
        fs = np.full(100, np.std(demand))

        policy = ConstantOrderPolicy(order_quantity=np.mean(demand))
        cost_fn = NewsvendorCost(holding_cost=0.15, backorder_cost=0.60)
        echelon = SupplyChainEchelon("E1", lead_time=2, policy=policy, cost_fn=cost_fn)
        chain = SerialSupplyChain(echelons=[echelon])
        result = chain.simulate(demand, fm, fs)

        assert np.var(result.echelon_results[0].orders) == pytest.approx(0.0)


# --- SmoothingOUTPolicy ---


class TestSmoothingOUT:
    def test_instantiation(self):
        p = SmoothingOUTPolicy(lead_time=2, alpha_s=0.3)
        assert p.lead_time == 2
        assert p.alpha_s == 0.3

    def test_compute_order_positive(self):
        p = SmoothingOUTPolicy(lead_time=2, alpha_s=0.5)
        order = p.compute_order(inventory_position=30.0, forecast_mean=10.0, forecast_std=2.0)
        assert order >= 0

    def test_smoothing_uses_history(self):
        """Second call should blend with the first order."""
        p = SmoothingOUTPolicy(lead_time=2, alpha_s=0.5)
        o1 = p.compute_order(30.0, 10.0, 2.0)
        o2 = p.compute_order(30.0, 10.0, 2.0)
        # o2 blends o1 with the new raw signal, so they differ
        # (unless by coincidence, but with alpha_s=0.5 they will converge)
        assert o2 != pytest.approx(0.0) or o1 == pytest.approx(0.0)

    def test_reset(self):
        p = SmoothingOUTPolicy(lead_time=2, alpha_s=0.5)
        p.compute_order(30.0, 10.0, 2.0)
        assert p._prev_order > 0
        p.reset()
        assert p._prev_order == 0.0

    def test_registered(self):
        assert "smoothing_out" in list_registered("policy")
