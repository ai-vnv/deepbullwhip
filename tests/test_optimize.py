"""Tests for optimization modules (inventory, policy tuning, network design).

Tests that require Pyomo solvers are skipped if no solver is available.
Policy tuning tests use simulation-based optimization which doesn't
require a solver.
"""

import numpy as np
import pytest

from deepbullwhip.chain.config import (
    EchelonConfig,
    consumer_2tier_config,
)
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph, from_serial


# ── Policy Tuning Tests (no solver required) ────────────────────────


class TestTuneServiceLevels:
    def test_basic_tuning(self):
        from deepbullwhip.optimize.policy_tuning import tune_service_levels

        graph = from_serial(consumer_2tier_config())
        rng = np.random.default_rng(42)
        scenarios = np.maximum(rng.normal(10.0, 2.0, (5, 26)), 0)

        result = tune_service_levels(
            graph, scenarios, objective="total_cost", grid_points=3
        )
        assert set(result.parameters.keys()) == {"Retailer", "Manufacturer"}
        assert all(0.80 <= v <= 0.99 for v in result.parameters.values())
        assert result.objective_value > 0
        assert result.n_evaluations > 0

    def test_bullwhip_objective(self):
        from deepbullwhip.optimize.policy_tuning import tune_service_levels

        graph = from_serial(consumer_2tier_config())
        rng = np.random.default_rng(42)
        scenarios = np.maximum(rng.normal(10.0, 2.0, (5, 26)), 0)

        result = tune_service_levels(
            graph, scenarios, objective="bullwhip", grid_points=3
        )
        assert result.objective_value > 0

    def test_weighted_objective(self):
        from deepbullwhip.optimize.policy_tuning import tune_service_levels

        graph = from_serial(consumer_2tier_config())
        rng = np.random.default_rng(42)
        scenarios = np.maximum(rng.normal(10.0, 2.0, (5, 26)), 0)

        result = tune_service_levels(
            graph, scenarios, objective="weighted", grid_points=3
        )
        assert result.objective_value > 0

    def test_1d_demand(self):
        from deepbullwhip.optimize.policy_tuning import tune_service_levels

        graph = from_serial(consumer_2tier_config())
        scenario = np.maximum(np.random.default_rng(0).normal(10, 2, 26), 0)

        result = tune_service_levels(
            graph, scenario, grid_points=3
        )
        assert len(result.parameters) == 2

    def test_coordinate_descent_for_large_network(self):
        """Networks with >3 nodes use coordinate descent."""
        from deepbullwhip.optimize.policy_tuning import tune_service_levels

        nodes = {
            f"E{i}": EchelonConfig(f"E{i}", lead_time=1, holding_cost=0.1, backorder_cost=0.5)
            for i in range(4)
        }
        edges = {
            (f"E{i+1}", f"E{i}"): EdgeConfig() for i in range(3)
        }
        graph = SupplyChainGraph(nodes=nodes, edges=edges)

        rng = np.random.default_rng(42)
        scenarios = np.maximum(rng.normal(10.0, 2.0, (3, 20)), 0)

        result = tune_service_levels(graph, scenarios, grid_points=3)
        assert len(result.parameters) == 4


class TestTuneSmoothingFactors:
    def test_basic_tuning(self):
        from deepbullwhip.optimize.policy_tuning import tune_smoothing_factors

        graph = from_serial(consumer_2tier_config())
        rng = np.random.default_rng(42)
        scenarios = np.maximum(rng.normal(10.0, 2.0, (5, 26)), 0)

        result = tune_smoothing_factors(graph, scenarios, grid_points=3)
        assert set(result.parameters.keys()) == {"Retailer", "Manufacturer"}
        assert all(0.1 <= v <= 1.0 for v in result.parameters.values())
        assert result.objective_value > 0

    def test_1d_demand(self):
        from deepbullwhip.optimize.policy_tuning import tune_smoothing_factors

        graph = from_serial(consumer_2tier_config())
        scenario = np.maximum(np.random.default_rng(0).normal(10, 2, 26), 0)

        result = tune_smoothing_factors(graph, scenario, grid_points=3)
        assert len(result.parameters) == 2


# ── Inventory Optimization Tests (require Pyomo + solver) ───────────


class TestBuildInventoryModel:
    def test_model_creation(self):
        pyomo = pytest.importorskip("pyomo")

        from deepbullwhip.optimize.inventory import build_inventory_model

        graph = from_serial(consumer_2tier_config())
        rng = np.random.default_rng(42)
        scenarios = np.maximum(rng.normal(10.0, 2.0, (5, 20)), 0)

        model = build_inventory_model(graph, scenarios)
        assert model is not None
        assert hasattr(model, "S")
        assert hasattr(model, "total_cost")

    def test_1d_demand(self):
        pyomo = pytest.importorskip("pyomo")

        from deepbullwhip.optimize.inventory import build_inventory_model

        graph = from_serial(consumer_2tier_config())
        scenario = np.maximum(np.random.default_rng(0).normal(10, 2, 20), 0)

        model = build_inventory_model(graph, scenario)
        assert model is not None

    def test_custom_service_levels(self):
        pyomo = pytest.importorskip("pyomo")

        from deepbullwhip.optimize.inventory import build_inventory_model

        graph = from_serial(consumer_2tier_config())
        rng = np.random.default_rng(42)
        scenarios = np.maximum(rng.normal(10.0, 2.0, (3, 15)), 0)

        svc = {"Retailer": 0.99, "Manufacturer": 0.90}
        model = build_inventory_model(graph, scenarios, service_levels=svc)
        assert model is not None


# ── Network Design Tests (require Pyomo + solver) ───────────────────


class TestBuildNetworkDesignModel:
    def test_model_creation(self):
        pyomo = pytest.importorskip("pyomo")

        from deepbullwhip.optimize.network_design import build_network_design_model

        candidates = {
            "F_A": EchelonConfig("F_A", 4, 0.10, 0.40),
            "F_B": EchelonConfig("F_B", 6, 0.08, 0.35),
            "WH": EchelonConfig("WH", 2, 0.15, 0.50),
            "Retail": EchelonConfig("Retail", 1, 0.20, 0.60),
        }
        edges = {
            ("F_A", "WH"): EdgeConfig(lead_time=2, transport_cost=0.05),
            ("F_B", "WH"): EdgeConfig(lead_time=3, transport_cost=0.03),
            ("WH", "Retail"): EdgeConfig(lead_time=1, transport_cost=0.02),
        }
        fixed_costs = {"F_A": 1000, "F_B": 800, "WH": 500, "Retail": 0}
        demand_volume = {"Retail": 100.0}

        model = build_network_design_model(
            candidates, edges, fixed_costs, demand_volume
        )
        assert model is not None
        assert hasattr(model, "open")
        assert hasattr(model, "flow")
        assert hasattr(model, "total_cost")


# ── Optional Dependency Tests ───────────────────────────────────────


class TestOptionalImport:
    def test_import_optional_installed(self):
        from deepbullwhip._optional import import_optional

        # numpy is always installed
        np_mod = import_optional("numpy", "core")
        assert np_mod is not None

    def test_import_optional_missing(self):
        from deepbullwhip._optional import import_optional

        with pytest.raises(ImportError, match="pip install"):
            import_optional("nonexistent_package_xyz", "fake")
