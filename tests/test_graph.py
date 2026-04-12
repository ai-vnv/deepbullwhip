"""Tests for SupplyChainGraph data model and from_serial conversion."""

import pytest

from deepbullwhip.chain.config import (
    EchelonConfig,
    beer_game_config,
    consumer_2tier_config,
    default_semiconductor_config,
)
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph, from_serial


class TestEdgeConfig:
    def test_defaults(self):
        edge = EdgeConfig()
        assert edge.lead_time == 1
        assert edge.capacity == float("inf")
        assert edge.transport_cost == 0.0

    def test_custom_values(self):
        edge = EdgeConfig(lead_time=3, capacity=100.0, transport_cost=0.5)
        assert edge.lead_time == 3
        assert edge.capacity == 100.0
        assert edge.transport_cost == 0.5


class TestSupplyChainGraph:
    def test_empty_graph(self):
        graph = SupplyChainGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_simple_graph(self):
        nodes = {
            "A": EchelonConfig("A", lead_time=1, holding_cost=0.1, backorder_cost=0.5),
            "B": EchelonConfig("B", lead_time=2, holding_cost=0.2, backorder_cost=0.6),
        }
        edges = {("A", "B"): EdgeConfig(lead_time=2)}
        graph = SupplyChainGraph(nodes=nodes, edges=edges)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_validation_missing_upstream(self):
        nodes = {"B": EchelonConfig("B", 1, 0.1, 0.5)}
        edges = {("A", "B"): EdgeConfig()}
        with pytest.raises(ValueError, match="upstream node 'A'"):
            SupplyChainGraph(nodes=nodes, edges=edges)

    def test_validation_missing_downstream(self):
        nodes = {"A": EchelonConfig("A", 1, 0.1, 0.5)}
        edges = {("A", "B"): EdgeConfig()}
        with pytest.raises(ValueError, match="downstream node 'B'"):
            SupplyChainGraph(nodes=nodes, edges=edges)

    def test_demand_nodes(self):
        nodes = {
            "Factory": EchelonConfig("Factory", 4, 0.1, 0.4),
            "Warehouse": EchelonConfig("Warehouse", 2, 0.15, 0.5),
            "Retail_A": EchelonConfig("Retail_A", 1, 0.2, 0.6),
            "Retail_B": EchelonConfig("Retail_B", 1, 0.2, 0.6),
        }
        edges = {
            ("Factory", "Warehouse"): EdgeConfig(),
            ("Warehouse", "Retail_A"): EdgeConfig(),
            ("Warehouse", "Retail_B"): EdgeConfig(),
        }
        graph = SupplyChainGraph(nodes=nodes, edges=edges)
        demand = graph.demand_nodes
        assert set(demand) == {"Retail_A", "Retail_B"}

    def test_source_nodes(self):
        nodes = {
            "Factory": EchelonConfig("Factory", 4, 0.1, 0.4),
            "Retailer": EchelonConfig("Retailer", 1, 0.2, 0.6),
        }
        edges = {("Factory", "Retailer"): EdgeConfig()}
        graph = SupplyChainGraph(nodes=nodes, edges=edges)
        assert graph.source_nodes == ["Factory"]

    def test_downstream_neighbors(self):
        nodes = {
            "A": EchelonConfig("A", 1, 0.1, 0.5),
            "B": EchelonConfig("B", 1, 0.1, 0.5),
            "C": EchelonConfig("C", 1, 0.1, 0.5),
        }
        edges = {("A", "B"): EdgeConfig(), ("A", "C"): EdgeConfig()}
        graph = SupplyChainGraph(nodes=nodes, edges=edges)
        assert set(graph.downstream_neighbors("A")) == {"B", "C"}
        assert graph.downstream_neighbors("B") == []

    def test_upstream_neighbors(self):
        nodes = {
            "A": EchelonConfig("A", 1, 0.1, 0.5),
            "B": EchelonConfig("B", 1, 0.1, 0.5),
            "C": EchelonConfig("C", 1, 0.1, 0.5),
        }
        edges = {("A", "C"): EdgeConfig(), ("B", "C"): EdgeConfig()}
        graph = SupplyChainGraph(nodes=nodes, edges=edges)
        assert set(graph.upstream_neighbors("C")) == {"A", "B"}
        assert graph.upstream_neighbors("A") == []

    def test_topological_order(self):
        nodes = {
            "Factory": EchelonConfig("Factory", 4, 0.1, 0.4),
            "Warehouse": EchelonConfig("Warehouse", 2, 0.15, 0.5),
            "Retailer": EchelonConfig("Retailer", 1, 0.2, 0.6),
        }
        edges = {
            ("Factory", "Warehouse"): EdgeConfig(),
            ("Warehouse", "Retailer"): EdgeConfig(),
        }
        graph = SupplyChainGraph(nodes=nodes, edges=edges)
        order = graph.topological_order()
        assert order.index("Factory") < order.index("Warehouse")
        assert order.index("Warehouse") < order.index("Retailer")

    def test_topological_order_cycle_detection(self):
        nodes = {
            "A": EchelonConfig("A", 1, 0.1, 0.5),
            "B": EchelonConfig("B", 1, 0.1, 0.5),
        }
        # Manually bypass validation to test cycle detection
        graph = SupplyChainGraph.__new__(SupplyChainGraph)
        graph.nodes = nodes
        graph.edges = {("A", "B"): EdgeConfig(), ("B", "A"): EdgeConfig()}
        with pytest.raises(ValueError, match="cycle"):
            graph.topological_order()


class TestFromSerial:
    def test_beer_game(self):
        configs = beer_game_config()
        graph = from_serial(configs)
        assert len(graph.nodes) == 4
        assert len(graph.edges) == 3
        # First config (Retailer) is demand-facing
        assert "Retailer" in graph.demand_nodes
        # Last config (Factory) is source
        assert "Factory" in graph.source_nodes

    def test_consumer_2tier(self):
        configs = consumer_2tier_config()
        graph = from_serial(configs)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert "Retailer" in graph.demand_nodes
        assert "Manufacturer" in graph.source_nodes

    def test_semiconductor(self):
        configs = default_semiconductor_config()
        graph = from_serial(configs)
        assert len(graph.nodes) == 4
        assert len(graph.edges) == 3
        assert "Distributor" in graph.demand_nodes
        assert "Supplier" in graph.source_nodes

    def test_single_echelon(self):
        configs = [EchelonConfig("Solo", 1, 0.1, 0.5)]
        graph = from_serial(configs)
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0
        assert graph.demand_nodes == ["Solo"]
        assert graph.source_nodes == ["Solo"]

    def test_topological_order_matches_serial(self):
        configs = beer_game_config()
        graph = from_serial(configs)
        order = graph.topological_order()
        # Sources come first in topological order
        assert order[0] == "Factory"
        assert order[-1] == "Retailer"

    def test_edge_lead_times(self):
        configs = beer_game_config()
        graph = from_serial(configs)
        # Edges should carry lead times from upstream config
        for (upstream, _downstream), edge in graph.edges.items():
            cfg = graph.nodes[upstream]
            assert edge.lead_time == cfg.lead_time
