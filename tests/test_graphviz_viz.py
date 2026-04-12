"""Tests for Graphviz visualization functions."""

import numpy as np
import pytest

from deepbullwhip._types import EchelonResult, NetworkSimulationResult
from deepbullwhip.chain.config import EchelonConfig, beer_game_config
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph, from_serial

graphviz = pytest.importorskip("graphviz")


@pytest.fixture
def beer_game_graph():
    return from_serial(beer_game_config())


@pytest.fixture
def simple_network_result(beer_game_graph):
    """Fake NetworkSimulationResult for testing visualization."""
    T = 10
    node_results = {}
    for name in beer_game_graph.nodes:
        node_results[name] = EchelonResult(
            name=name,
            orders=np.random.default_rng(42).uniform(3, 6, T),
            inventory_levels=np.random.default_rng(42).uniform(-5, 50, T),
            costs=np.random.default_rng(42).uniform(0, 10, T),
            bullwhip_ratio=1.5,
            fill_rate=0.9,
            total_cost=50.0,
        )
    edge_flows = {
        edge: np.random.default_rng(42).uniform(2, 6, T)
        for edge in beer_game_graph.edges
    }
    return NetworkSimulationResult(
        node_results=node_results,
        edge_flows=edge_flows,
        cumulative_bullwhip=2.5,
        total_cost=200.0,
    )


class TestRenderNetwork:
    def test_basic_render(self, beer_game_graph):
        from deepbullwhip.diagnostics.graphviz_viz import render_network

        source = render_network(beer_game_graph)
        assert source is not None
        dot_text = source.source
        assert "digraph" in dot_text
        assert "Factory" in dot_text
        assert "Retailer" in dot_text

    def test_render_with_result(self, beer_game_graph, simple_network_result):
        from deepbullwhip.diagnostics.graphviz_viz import render_network

        source = render_network(beer_game_graph, sim_result=simple_network_result)
        dot_text = source.source
        assert "BW=" in dot_text
        assert "FR=" in dot_text

    def test_render_engines(self, beer_game_graph):
        from deepbullwhip.diagnostics.graphviz_viz import render_network

        for engine in ["dot", "neato", "fdp"]:
            source = render_network(beer_game_graph, engine=engine)
            assert source is not None

    def test_render_rankdir(self, beer_game_graph):
        from deepbullwhip.diagnostics.graphviz_viz import render_network

        source_lr = render_network(beer_game_graph, rankdir="LR")
        assert "rankdir=LR" in source_lr.source
        source_tb = render_network(beer_game_graph, rankdir="TB")
        assert "rankdir=TB" in source_tb.source

    def test_render_with_title(self, beer_game_graph):
        from deepbullwhip.diagnostics.graphviz_viz import render_network

        source = render_network(beer_game_graph, title="Beer Game")
        assert "Beer Game" in source.source

    def test_render_tree_topology(self):
        from deepbullwhip.diagnostics.graphviz_viz import render_network

        graph = SupplyChainGraph(
            nodes={
                "Factory": EchelonConfig("Factory", 4, 0.1, 0.4),
                "WH_A": EchelonConfig("WH_A", 2, 0.15, 0.5),
                "WH_B": EchelonConfig("WH_B", 2, 0.15, 0.5),
            },
            edges={
                ("Factory", "WH_A"): EdgeConfig(lead_time=3, capacity=100),
                ("Factory", "WH_B"): EdgeConfig(lead_time=2, transport_cost=0.5),
            },
        )
        source = render_network(graph)
        dot_text = source.source
        assert "cap=100" in dot_text
        assert "tc=0.50" in dot_text


class TestRenderSimulationSnapshot:
    def test_snapshot(self, beer_game_graph, simple_network_result):
        from deepbullwhip.diagnostics.graphviz_viz import render_simulation_snapshot

        source = render_simulation_snapshot(
            beer_game_graph, simple_network_result, period=5
        )
        dot_text = source.source
        assert "t=5" in dot_text
        assert "Inv=" in dot_text
        assert "Order=" in dot_text

    def test_snapshot_period_zero(self, beer_game_graph, simple_network_result):
        from deepbullwhip.diagnostics.graphviz_viz import render_simulation_snapshot

        source = render_simulation_snapshot(
            beer_game_graph, simple_network_result, period=0
        )
        assert source is not None


class TestSaveFigure:
    def test_unsupported_format(self, beer_game_graph):
        from deepbullwhip.diagnostics.graphviz_viz import render_network, save_figure

        source = render_network(beer_game_graph)
        with pytest.raises(ValueError, match="Unsupported format"):
            save_figure(source, "output.xyz")


class TestBwColorBranches:
    def test_low_bullwhip_color(self, beer_game_graph):
        """Test BWR < 1.5 (green) color branch."""
        from deepbullwhip.diagnostics.graphviz_viz import render_network

        T = 10
        node_results = {}
        for name in beer_game_graph.nodes:
            node_results[name] = EchelonResult(
                name=name,
                orders=np.full(T, 5.0),
                inventory_levels=np.full(T, 20.0),
                costs=np.full(T, 1.0),
                bullwhip_ratio=1.0,  # < 1.5 -> green
                fill_rate=0.95,
                total_cost=10.0,
            )
        result = NetworkSimulationResult(
            node_results=node_results,
            edge_flows={e: np.full(T, 5.0) for e in beer_game_graph.edges},
            cumulative_bullwhip=1.0,
            total_cost=40.0,
        )
        source = render_network(beer_game_graph, sim_result=result)
        assert "#2E8B57" in source.source  # green color

    def test_high_bullwhip_color(self, beer_game_graph):
        """Test BWR >= 3.0 (red) color branch."""
        from deepbullwhip.diagnostics.graphviz_viz import render_network

        T = 10
        node_results = {}
        for name in beer_game_graph.nodes:
            node_results[name] = EchelonResult(
                name=name,
                orders=np.full(T, 5.0),
                inventory_levels=np.full(T, 20.0),
                costs=np.full(T, 1.0),
                bullwhip_ratio=4.0,  # >= 3.0 -> red
                fill_rate=0.5,
                total_cost=10.0,
            )
        result = NetworkSimulationResult(
            node_results=node_results,
            edge_flows={e: np.full(T, 5.0) for e in beer_game_graph.edges},
            cumulative_bullwhip=4.0,
            total_cost=40.0,
        )
        source = render_network(beer_game_graph, sim_result=result)
        assert "#CD5C5C" in source.source  # red color

    def test_render_with_simulation_result(self, beer_game_graph):
        """Test rendering with SimulationResult (positional, not named)."""
        from deepbullwhip._types import SimulationResult
        from deepbullwhip.diagnostics.graphviz_viz import render_network

        T = 10
        echelon_results = [
            EchelonResult(
                name=name,
                orders=np.full(T, 5.0),
                inventory_levels=np.full(T, 20.0),
                costs=np.full(T, 1.0),
                bullwhip_ratio=2.0,
                fill_rate=0.9,
                total_cost=10.0,
            )
            for name in beer_game_graph.nodes
        ]
        sim_result = SimulationResult(
            echelon_results=echelon_results,
            cumulative_bullwhip=2.0,
            total_cost=40.0,
        )
        source = render_network(beer_game_graph, sim_result=sim_result)
        assert "BW=" in source.source


class TestSaveFigureExecution:
    def test_save_svg(self, beer_game_graph, tmp_path):
        import shutil

        if shutil.which("dot") is None:
            pytest.skip("Graphviz 'dot' executable not found on PATH")

        from deepbullwhip.diagnostics.graphviz_viz import render_network, save_figure

        source = render_network(beer_game_graph)
        outpath = str(tmp_path / "test_output.svg")
        result = save_figure(source, outpath)
        assert result is not None


class TestSupplyChainNetworkBridge:
    def test_from_graph(self, beer_game_graph):
        from deepbullwhip.diagnostics.network import SupplyChainNetwork

        network = SupplyChainNetwork.from_graph(beer_game_graph)
        assert len(network.nodes) == 4
        assert len(network.edges) == 3

    def test_from_graph_with_locations(self, beer_game_graph):
        from deepbullwhip.diagnostics.network import SupplyChainNetwork

        locations = {
            "Factory": (40.0, -74.0),
            "Retailer": (34.0, -118.0),
        }
        network = SupplyChainNetwork.from_graph(beer_game_graph, locations=locations)
        # Factory should have the provided coordinates
        factory_node = next(n for n in network.nodes if n.name == "Factory")
        assert factory_node.lat == 40.0
        assert factory_node.lon == -74.0
