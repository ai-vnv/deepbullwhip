"""Tests for NetworkX conversion functions."""

import pytest

from deepbullwhip.chain.config import (
    EchelonConfig,
    beer_game_config,
    consumer_2tier_config,
)
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph, from_serial

networkx = pytest.importorskip("networkx")


class TestToNetworkx:
    def test_beer_game_conversion(self):
        from deepbullwhip.network.convert import to_networkx

        graph = from_serial(beer_game_config())
        G = to_networkx(graph)
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 3
        assert networkx.is_directed_acyclic_graph(G)

    def test_node_attributes(self):
        from deepbullwhip.network.convert import to_networkx

        graph = SupplyChainGraph(
            nodes={"X": EchelonConfig("X", lead_time=3, holding_cost=0.2, backorder_cost=0.8)},
            edges={},
        )
        G = to_networkx(graph)
        attrs = G.nodes["X"]
        assert attrs["lead_time"] == 3
        assert attrs["holding_cost"] == 0.2
        assert attrs["backorder_cost"] == 0.8
        assert attrs["service_level"] == 0.95
        assert attrs["initial_inventory"] == 50.0

    def test_edge_attributes(self):
        from deepbullwhip.network.convert import to_networkx

        graph = SupplyChainGraph(
            nodes={
                "A": EchelonConfig("A", 1, 0.1, 0.5),
                "B": EchelonConfig("B", 1, 0.1, 0.5),
            },
            edges={("A", "B"): EdgeConfig(lead_time=5, capacity=200, transport_cost=0.3)},
        )
        G = to_networkx(graph)
        attrs = G.edges["A", "B"]
        assert attrs["lead_time"] == 5
        assert attrs["capacity"] == 200
        assert attrs["transport_cost"] == 0.3

    def test_tree_topology(self):
        from deepbullwhip.network.convert import to_networkx

        nodes = {
            "Factory": EchelonConfig("Factory", 4, 0.1, 0.4),
            "WH_East": EchelonConfig("WH_East", 2, 0.15, 0.5),
            "WH_West": EchelonConfig("WH_West", 2, 0.15, 0.5),
            "Retail": EchelonConfig("Retail", 1, 0.2, 0.6),
        }
        edges = {
            ("Factory", "WH_East"): EdgeConfig(),
            ("Factory", "WH_West"): EdgeConfig(),
            ("WH_East", "Retail"): EdgeConfig(),
        }
        graph = SupplyChainGraph(nodes=nodes, edges=edges)
        G = to_networkx(graph)
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 3


class TestFromNetworkx:
    def test_round_trip(self):
        from deepbullwhip.network.convert import from_networkx, to_networkx

        original = from_serial(beer_game_config())
        G = to_networkx(original)
        restored = from_networkx(G)
        assert set(restored.nodes.keys()) == set(original.nodes.keys())
        assert set(restored.edges.keys()) == set(original.edges.keys())

    def test_from_raw_networkx_graph(self):
        from deepbullwhip.network.convert import from_networkx

        G = networkx.DiGraph()
        G.add_node("Factory", lead_time=4, holding_cost=0.1, backorder_cost=0.4)
        G.add_node("Retailer", lead_time=1, holding_cost=0.2, backorder_cost=0.6)
        G.add_edge("Factory", "Retailer", lead_time=2)
        graph = from_networkx(G)
        assert "Factory" in graph.nodes
        assert "Retailer" in graph.nodes
        assert ("Factory", "Retailer") in graph.edges

    def test_defaults_for_missing_attributes(self):
        from deepbullwhip.network.convert import from_networkx

        G = networkx.DiGraph()
        G.add_node("X")
        graph = from_networkx(G)
        cfg = graph.nodes["X"]
        assert cfg.lead_time == 1
        assert cfg.holding_cost == 0.10
        assert cfg.backorder_cost == 0.40
        assert cfg.service_level == 0.95

    def test_rejects_cyclic_graph(self):
        from deepbullwhip.network.convert import from_networkx

        G = networkx.DiGraph()
        G.add_edge("A", "B")
        G.add_edge("B", "A")
        with pytest.raises(ValueError, match="DAG"):
            from_networkx(G)

    def test_attribute_preservation_round_trip(self):
        from deepbullwhip.network.convert import from_networkx, to_networkx

        original = SupplyChainGraph(
            nodes={
                "F": EchelonConfig("F", lead_time=8, holding_cost=0.05, backorder_cost=0.3,
                                   depreciation_rate=0.01, service_level=0.90, initial_inventory=100),
            },
            edges={},
        )
        G = to_networkx(original)
        restored = from_networkx(G)
        cfg = restored.nodes["F"]
        assert cfg.lead_time == 8
        assert cfg.holding_cost == 0.05
        assert cfg.backorder_cost == 0.3
        assert cfg.depreciation_rate == 0.01
        assert cfg.service_level == 0.90
        assert cfg.initial_inventory == 100


class TestSerialToNetworkx:
    def test_convenience_function(self):
        from deepbullwhip.network.convert import serial_to_networkx

        G = serial_to_networkx(beer_game_config())
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 3
        assert networkx.is_directed_acyclic_graph(G)

    def test_consumer_2tier(self):
        from deepbullwhip.network.convert import serial_to_networkx

        G = serial_to_networkx(consumer_2tier_config())
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
