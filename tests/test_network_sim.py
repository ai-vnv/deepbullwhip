"""Tests for NetworkSupplyChain simulator."""

import numpy as np
import pytest

from deepbullwhip._types import NetworkSimulationResult
from deepbullwhip.chain.config import (
    EchelonConfig,
    beer_game_config,
    consumer_2tier_config,
)
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph, from_serial
from deepbullwhip.chain.network_sim import NetworkSupplyChain


@pytest.fixture
def beer_game_network():
    return NetworkSupplyChain(from_serial(beer_game_config()))


@pytest.fixture
def simple_demand():
    return np.full(52, 4.0)


@pytest.fixture
def tree_graph():
    return SupplyChainGraph(
        nodes={
            "Factory": EchelonConfig("Factory", 4, 0.10, 0.40),
            "Warehouse": EchelonConfig("Warehouse", 2, 0.15, 0.50),
            "Retail_A": EchelonConfig("Retail_A", 1, 0.20, 0.60),
            "Retail_B": EchelonConfig("Retail_B", 1, 0.20, 0.60),
        },
        edges={
            ("Factory", "Warehouse"): EdgeConfig(lead_time=3),
            ("Warehouse", "Retail_A"): EdgeConfig(lead_time=1),
            ("Warehouse", "Retail_B"): EdgeConfig(lead_time=1),
        },
    )


class TestNetworkSupplyChainInit:
    def test_from_serial_config(self):
        chain = NetworkSupplyChain(from_serial(beer_game_config()))
        assert len(chain._echelons) == 4

    def test_from_serial_classmethod(self):
        chain = NetworkSupplyChain.from_serial(beer_game_config())
        assert len(chain._echelons) == 4

    def test_tree_topology(self, tree_graph):
        chain = NetworkSupplyChain(tree_graph)
        assert len(chain._echelons) == 4

    def test_custom_policies(self):
        from deepbullwhip.policy.constant_order import ConstantOrderPolicy

        graph = from_serial(consumer_2tier_config())
        policies = {
            "Retailer": ConstantOrderPolicy(order_quantity=10.0),
            "Manufacturer": ConstantOrderPolicy(order_quantity=10.0),
        }
        chain = NetworkSupplyChain(graph, policies=policies)
        assert len(chain._echelons) == 2


class TestNetworkSimulation:
    def test_serial_simulation_returns_result(self, beer_game_network, simple_demand):
        result = beer_game_network.simulate(
            demand={"Retailer": simple_demand},
            forecasts_mean={"Retailer": np.full(52, 4.0)},
            forecasts_std={"Retailer": np.full(52, 1.0)},
        )
        assert isinstance(result, NetworkSimulationResult)

    def test_result_has_all_nodes(self, beer_game_network, simple_demand):
        result = beer_game_network.simulate(
            demand={"Retailer": simple_demand},
            forecasts_mean={"Retailer": np.full(52, 4.0)},
            forecasts_std={"Retailer": np.full(52, 1.0)},
        )
        assert set(result.node_results.keys()) == {
            "Retailer", "Wholesaler", "Distributor", "Factory"
        }

    def test_result_shapes(self, beer_game_network, simple_demand):
        T = len(simple_demand)
        result = beer_game_network.simulate(
            demand={"Retailer": simple_demand},
            forecasts_mean={"Retailer": np.full(T, 4.0)},
            forecasts_std={"Retailer": np.full(T, 1.0)},
        )
        for name, er in result.node_results.items():
            assert er.orders.shape == (T,), f"{name} orders shape mismatch"
            assert er.inventory_levels.shape == (T,), f"{name} inv shape mismatch"
            assert er.costs.shape == (T,), f"{name} costs shape mismatch"

    def test_fill_rate_bounded(self, beer_game_network, simple_demand):
        result = beer_game_network.simulate(
            demand={"Retailer": simple_demand},
            forecasts_mean={"Retailer": np.full(52, 4.0)},
            forecasts_std={"Retailer": np.full(52, 1.0)},
        )
        for er in result.node_results.values():
            assert 0 <= er.fill_rate <= 1

    def test_bullwhip_ratio_positive(self, beer_game_network, simple_demand):
        result = beer_game_network.simulate(
            demand={"Retailer": simple_demand},
            forecasts_mean={"Retailer": np.full(52, 4.0)},
            forecasts_std={"Retailer": np.full(52, 1.0)},
        )
        for er in result.node_results.values():
            assert er.bullwhip_ratio > 0

    def test_total_cost_is_sum(self, beer_game_network, simple_demand):
        result = beer_game_network.simulate(
            demand={"Retailer": simple_demand},
            forecasts_mean={"Retailer": np.full(52, 4.0)},
            forecasts_std={"Retailer": np.full(52, 1.0)},
        )
        expected = sum(er.total_cost for er in result.node_results.values())
        assert result.total_cost == pytest.approx(expected)

    def test_missing_demand_raises(self, beer_game_network, simple_demand):
        with pytest.raises(ValueError, match="Missing demand"):
            beer_game_network.simulate(
                demand={},
                forecasts_mean={"Retailer": np.full(52, 4.0)},
                forecasts_std={"Retailer": np.full(52, 1.0)},
            )

    def test_to_dict(self, beer_game_network, simple_demand):
        result = beer_game_network.simulate(
            demand={"Retailer": simple_demand},
            forecasts_mean={"Retailer": np.full(52, 4.0)},
            forecasts_std={"Retailer": np.full(52, 1.0)},
        )
        d = result.to_dict()
        assert "BW_Retailer" in d
        assert "cost_Retailer" in d
        assert "fill_rate_Retailer" in d
        assert "BW_cumulative" in d
        assert "total_cost" in d

    def test_to_simulation_result(self, beer_game_network, simple_demand):
        result = beer_game_network.simulate(
            demand={"Retailer": simple_demand},
            forecasts_mean={"Retailer": np.full(52, 4.0)},
            forecasts_std={"Retailer": np.full(52, 1.0)},
        )
        sim_result = result.to_simulation_result()
        assert len(sim_result.echelon_results) == 4
        assert sim_result.total_cost == result.total_cost


class TestTreeTopologySimulation:
    def test_tree_simulation(self, tree_graph):
        chain = NetworkSupplyChain(tree_graph)
        T = 30
        demand = {
            "Retail_A": np.full(T, 5.0),
            "Retail_B": np.full(T, 3.0),
        }
        fm = {
            "Retail_A": np.full(T, 5.0),
            "Retail_B": np.full(T, 3.0),
        }
        fs = {
            "Retail_A": np.full(T, 1.0),
            "Retail_B": np.full(T, 1.0),
        }
        result = chain.simulate(demand, fm, fs)
        assert len(result.node_results) == 4
        assert "Factory" in result.node_results
        assert "Warehouse" in result.node_results

    def test_tree_edge_flows(self, tree_graph):
        chain = NetworkSupplyChain(tree_graph)
        T = 20
        demand = {
            "Retail_A": np.full(T, 5.0),
            "Retail_B": np.full(T, 3.0),
        }
        fm = {
            "Retail_A": np.full(T, 5.0),
            "Retail_B": np.full(T, 3.0),
        }
        fs = {
            "Retail_A": np.full(T, 1.0),
            "Retail_B": np.full(T, 1.0),
        }
        result = chain.simulate(demand, fm, fs)
        assert len(result.edge_flows) > 0


class TestFromNetworkx:
    def test_from_networkx_classmethod(self):
        pytest.importorskip("networkx")
        import networkx as nx

        G = nx.DiGraph()
        G.add_node("A", lead_time=1, holding_cost=0.1, backorder_cost=0.5)
        G.add_node("B", lead_time=2, holding_cost=0.2, backorder_cost=0.6)
        G.add_edge("B", "A", lead_time=2)

        chain = NetworkSupplyChain.from_networkx(G)
        assert len(chain._echelons) == 2

    def test_from_networkx_simulation(self):
        pytest.importorskip("networkx")
        import networkx as nx

        G = nx.DiGraph()
        G.add_node("Supplier", lead_time=3, holding_cost=0.1, backorder_cost=0.4)
        G.add_node("Store", lead_time=1, holding_cost=0.2, backorder_cost=0.6)
        G.add_edge("Supplier", "Store", lead_time=2)

        chain = NetworkSupplyChain.from_networkx(G)
        T = 20
        result = chain.simulate(
            demand={"Store": np.full(T, 5.0)},
            forecasts_mean={"Store": np.full(T, 5.0)},
            forecasts_std={"Store": np.full(T, 1.0)},
        )
        assert isinstance(result, NetworkSimulationResult)
        assert "Store" in result.node_results
        assert "Supplier" in result.node_results
