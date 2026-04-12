"""Tests for NetworkX-based supply chain graph analysis."""

import pytest

from deepbullwhip.chain.config import beer_game_config, EchelonConfig
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph, from_serial

networkx = pytest.importorskip("networkx")


@pytest.fixture
def beer_game_graph():
    from deepbullwhip.network.convert import to_networkx
    return to_networkx(from_serial(beer_game_config()))


@pytest.fixture
def tree_graph():
    from deepbullwhip.network.convert import to_networkx

    graph = SupplyChainGraph(
        nodes={
            "Factory": EchelonConfig("Factory", 4, 0.1, 0.4),
            "WH_A": EchelonConfig("WH_A", 2, 0.15, 0.5),
            "WH_B": EchelonConfig("WH_B", 3, 0.15, 0.5),
            "Retail": EchelonConfig("Retail", 1, 0.2, 0.6),
        },
        edges={
            ("Factory", "WH_A"): EdgeConfig(lead_time=5),
            ("Factory", "WH_B"): EdgeConfig(lead_time=3),
            ("WH_A", "Retail"): EdgeConfig(lead_time=2),
        },
    )
    return to_networkx(graph)


class TestFindCriticalPath:
    def test_serial_chain(self, beer_game_graph):
        from deepbullwhip.network.analysis import find_critical_path

        path = find_critical_path(beer_game_graph)
        assert len(path) == 4
        assert path[0] == "Factory"
        assert path[-1] == "Retailer"

    def test_tree_critical_path(self, tree_graph):
        from deepbullwhip.network.analysis import find_critical_path

        path = find_critical_path(tree_graph)
        assert "Factory" in path
        assert len(path) >= 2


class TestCriticalPathLength:
    def test_serial_chain(self, beer_game_graph):
        from deepbullwhip.network.analysis import critical_path_length

        length = critical_path_length(beer_game_graph)
        assert length > 0
        assert isinstance(length, (int, float))


class TestEchelonCentrality:
    def test_returns_dict(self, beer_game_graph):
        from deepbullwhip.network.analysis import echelon_centrality

        centrality = echelon_centrality(beer_game_graph)
        assert isinstance(centrality, dict)
        assert len(centrality) == 4
        for v in centrality.values():
            assert 0 <= v <= 1

    def test_tree_centrality(self, tree_graph):
        from deepbullwhip.network.analysis import echelon_centrality

        centrality = echelon_centrality(tree_graph)
        assert isinstance(centrality, dict)


class TestUpstreamDownstream:
    def test_upstream_retailer(self, beer_game_graph):
        from deepbullwhip.network.analysis import upstream_nodes

        ancestors = upstream_nodes(beer_game_graph, "Retailer")
        assert "Factory" in ancestors
        assert "Wholesaler" in ancestors
        assert "Distributor" in ancestors
        assert "Retailer" not in ancestors

    def test_downstream_factory(self, beer_game_graph):
        from deepbullwhip.network.analysis import downstream_nodes

        descendants = downstream_nodes(beer_game_graph, "Factory")
        assert "Retailer" in descendants
        assert "Factory" not in descendants

    def test_source_has_no_upstream(self, beer_game_graph):
        from deepbullwhip.network.analysis import upstream_nodes

        assert upstream_nodes(beer_game_graph, "Factory") == set()

    def test_demand_has_no_downstream(self, beer_game_graph):
        from deepbullwhip.network.analysis import downstream_nodes

        assert downstream_nodes(beer_game_graph, "Retailer") == set()


class TestTopologicalOrder:
    def test_serial_order(self, beer_game_graph):
        from deepbullwhip.network.analysis import topological_order

        order = topological_order(beer_game_graph)
        assert len(order) == 4
        assert order.index("Factory") < order.index("Retailer")

    def test_tree_order(self, tree_graph):
        from deepbullwhip.network.analysis import topological_order

        order = topological_order(tree_graph)
        assert order[0] == "Factory"  # Source first
