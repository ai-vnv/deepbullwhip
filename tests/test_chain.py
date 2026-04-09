import numpy as np
import pytest

from deepbullwhip import (
    EchelonConfig,
    NewsvendorCost,
    OrderUpToPolicy,
    SemiconductorDemandGenerator,
    SerialSupplyChain,
    SupplyChainEchelon,
)
from deepbullwhip._types import SimulationResult


class TestSupplyChainEchelon:
    def test_reset_clears_state(self, simple_echelon):
        simple_echelon.reset()
        simple_echelon.step(10.0, 10.0, 1.0)
        assert len(simple_echelon.orders) == 1

        simple_echelon.reset()
        assert len(simple_echelon.orders) == 0
        assert len(simple_echelon.inventory_levels) == 0
        assert len(simple_echelon.costs) == 0
        assert simple_echelon.inventory == 50.0

    def test_pipeline_length_after_reset(self, simple_echelon):
        simple_echelon.reset()
        assert len(simple_echelon.pipeline) == simple_echelon.lead_time

    def test_step_returns_nonnegative_order(self, simple_echelon):
        simple_echelon.reset()
        order = simple_echelon.step(10.0, 10.0, 1.0)
        assert order >= 0.0

    def test_arrays_property(self, simple_echelon):
        simple_echelon.reset()
        for _ in range(5):
            simple_echelon.step(10.0, 10.0, 1.0)
        assert simple_echelon.orders_array.shape == (5,)
        assert simple_echelon.inventory_array.shape == (5,)
        assert simple_echelon.costs_array.shape == (5,)


class TestSerialSupplyChain:
    def test_default_chain_has_4_echelons(self, default_chain):
        assert default_chain.K == 4
        assert len(default_chain.echelons) == 4

    def test_echelon_names(self, default_chain):
        names = [e.name for e in default_chain.echelons]
        assert names == ["Distributor", "OSAT", "Foundry", "Supplier"]

    def test_simulation_returns_result(self, default_chain, demand_series):
        fm = np.full_like(demand_series, demand_series.mean())
        fs = np.full_like(demand_series, demand_series.std())
        result = default_chain.simulate(demand_series, fm, fs)
        assert isinstance(result, SimulationResult)

    def test_simulation_result_shapes(self, default_chain, demand_series):
        T = len(demand_series)
        fm = np.full_like(demand_series, demand_series.mean())
        fs = np.full_like(demand_series, demand_series.std())
        result = default_chain.simulate(demand_series, fm, fs)

        for er in result.echelon_results:
            assert er.orders.shape == (T,)
            assert er.inventory_levels.shape == (T,)
            assert er.costs.shape == (T,)

    def test_bullwhip_ratio_positive(self, default_chain, demand_series):
        fm = np.full_like(demand_series, demand_series.mean())
        fs = np.full_like(demand_series, demand_series.std())
        result = default_chain.simulate(demand_series, fm, fs)

        for er in result.echelon_results:
            assert er.bullwhip_ratio > 0

    def test_fill_rate_bounded(self, default_chain, demand_series):
        fm = np.full_like(demand_series, demand_series.mean())
        fs = np.full_like(demand_series, demand_series.std())
        result = default_chain.simulate(demand_series, fm, fs)

        for er in result.echelon_results:
            assert 0 <= er.fill_rate <= 1

    def test_total_cost_is_sum(self, default_chain, demand_series):
        fm = np.full_like(demand_series, demand_series.mean())
        fs = np.full_like(demand_series, demand_series.std())
        result = default_chain.simulate(demand_series, fm, fs)

        expected = sum(er.total_cost for er in result.echelon_results)
        assert result.total_cost == pytest.approx(expected)

    def test_to_dict_keys(self, default_chain, demand_series):
        fm = np.full_like(demand_series, demand_series.mean())
        fs = np.full_like(demand_series, demand_series.std())
        result = default_chain.simulate(demand_series, fm, fs)
        d = result.to_dict()

        for k in range(1, 5):
            assert f"BW_{k}" in d
            assert f"cost_{k}" in d
            assert f"fill_rate_{k}" in d
        assert "BW_cumulative" in d
        assert "total_cost" in d

    def test_custom_chain_from_config(self):
        configs = [
            EchelonConfig("A", lead_time=1, holding_cost=0.1, backorder_cost=0.5),
            EchelonConfig("B", lead_time=2, holding_cost=0.2, backorder_cost=0.6),
        ]
        chain = SerialSupplyChain.from_config(configs)
        assert chain.K == 2
        assert chain.echelons[0].name == "A"
        assert chain.echelons[1].name == "B"

    def test_constant_demand_low_bullwhip(self, constant_demand):
        """With constant demand and perfect forecast, BW should be near 1.0."""
        configs = [
            EchelonConfig("E1", lead_time=1, holding_cost=0.1, backorder_cost=0.5),
        ]
        chain = SerialSupplyChain.from_config(configs)
        fm = np.full_like(constant_demand, 10.0)
        fs = np.zeros_like(constant_demand)
        result = chain.simulate(constant_demand, fm, fs)
        # With constant demand and perfect forecast, bullwhip should be very low
        # Allow some tolerance for transient startup effects
        assert result.echelon_results[0].bullwhip_ratio < 2.0
