"""Tests for multi-backend rendering (matplotlib, graphviz, tikz) and unified API."""

import numpy as np
import pytest

from deepbullwhip._types import EchelonResult, NetworkSimulationResult
from deepbullwhip.chain.config import (
    EchelonConfig,
    beer_game_config,
    consumer_2tier_config,
)
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph, from_serial
from deepbullwhip.render.api import render_graph
from deepbullwhip.render.theme import get_theme
from deepbullwhip.schema.definition import NodeLayoutHint


@pytest.fixture
def beer_game_graph():
    return from_serial(beer_game_config())


@pytest.fixture
def consumer_graph():
    return from_serial(consumer_2tier_config())


@pytest.fixture
def tree_graph():
    return SupplyChainGraph(
        nodes={
            "Factory": EchelonConfig("Factory", 4, 0.10, 0.40),
            "WH": EchelonConfig("WH", 2, 0.15, 0.50),
            "Store_A": EchelonConfig("Store_A", 1, 0.20, 0.60),
            "Store_B": EchelonConfig("Store_B", 1, 0.20, 0.60),
        },
        edges={
            ("Factory", "WH"): EdgeConfig(lead_time=3),
            ("WH", "Store_A"): EdgeConfig(lead_time=1),
            ("WH", "Store_B"): EdgeConfig(lead_time=1),
        },
    )


@pytest.fixture
def fake_result(beer_game_graph):
    T = 10
    node_results = {}
    for name in beer_game_graph.nodes:
        node_results[name] = EchelonResult(
            name=name,
            orders=np.full(T, 5.0),
            inventory_levels=np.full(T, 20.0),
            costs=np.full(T, 1.0),
            bullwhip_ratio=1.5,
            fill_rate=0.9,
            total_cost=10.0,
        )
    return NetworkSimulationResult(
        node_results=node_results,
        edge_flows={e: np.full(T, 5.0) for e in beer_game_graph.edges},
        cumulative_bullwhip=2.0,
        total_cost=40.0,
    )


# ── Matplotlib Backend Tests ────────────────────────────────────────


class TestMatplotlibBackend:
    def test_basic_render(self, beer_game_graph):
        import matplotlib.figure

        fig = render_graph(beer_game_graph, backend="matplotlib", theme="kfupm")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt = __import__("matplotlib.pyplot", fromlist=["close"])
        plt.close(fig)

    def test_all_themes(self, consumer_graph):
        import matplotlib.pyplot as plt

        for theme_name in ["kfupm", "ieee", "presentation", "minimal"]:
            fig = render_graph(consumer_graph, backend="matplotlib", theme=theme_name)
            assert fig is not None
            plt.close(fig)

    def test_with_sim_result(self, beer_game_graph, fake_result):
        import matplotlib.pyplot as plt

        fig = render_graph(
            beer_game_graph,
            backend="matplotlib",
            sim_result=fake_result,
            title="Test",
        )
        assert fig is not None
        plt.close(fig)

    def test_with_annotations(self, consumer_graph):
        import matplotlib.pyplot as plt

        fig = render_graph(
            consumer_graph,
            backend="matplotlib",
            annotations={"Retailer": {"Status": "Active"}},
        )
        assert fig is not None
        plt.close(fig)

    def test_tree_topology(self, tree_graph):
        import matplotlib.pyplot as plt

        fig = render_graph(tree_graph, backend="matplotlib")
        assert fig is not None
        plt.close(fig)

    def test_with_layout_hints(self, consumer_graph):
        import matplotlib.pyplot as plt

        hints = {"Retailer": NodeLayoutHint(position=(0, 0))}
        fig = render_graph(
            consumer_graph, backend="matplotlib", layout_hints=hints
        )
        assert fig is not None
        plt.close(fig)

    def test_custom_theme(self, consumer_graph):
        import matplotlib.pyplot as plt
        from deepbullwhip.render.theme import FontStyle

        theme = get_theme("kfupm").override(font=FontStyle(node_label_size=14.0))
        fig = render_graph(consumer_graph, backend="matplotlib", theme=theme)
        assert fig is not None
        plt.close(fig)


# ── TikZ Backend Tests ──────────────────────────────────────────────


class TestTikzBackend:
    def test_basic_render(self, beer_game_graph):
        tex = render_graph(beer_game_graph, backend="tikz", theme="kfupm")
        assert isinstance(tex, str)
        assert r"\begin{tikzpicture}" in tex
        assert r"\end{tikzpicture}" in tex

    def test_standalone_mode(self, consumer_graph):
        tex = render_graph(consumer_graph, backend="tikz", standalone=True)
        assert r"\documentclass" in tex
        assert r"\begin{document}" in tex

    def test_fragment_mode(self, consumer_graph):
        tex = render_graph(consumer_graph, backend="tikz", standalone=False)
        assert r"\documentclass" not in tex
        assert r"\begin{tikzpicture}" in tex

    def test_all_themes(self, consumer_graph):
        for theme_name in ["kfupm", "ieee", "presentation", "minimal"]:
            tex = render_graph(consumer_graph, backend="tikz", theme=theme_name)
            assert r"\begin{tikzpicture}" in tex

    def test_with_title(self, beer_game_graph):
        tex = render_graph(
            beer_game_graph, backend="tikz", title="Beer Game"
        )
        assert "Beer Game" in tex

    def test_with_sim_result(self, beer_game_graph, fake_result):
        tex = render_graph(
            beer_game_graph, backend="tikz", sim_result=fake_result
        )
        assert "BW=" in tex

    def test_tree_topology(self, tree_graph):
        tex = render_graph(tree_graph, backend="tikz")
        assert "Factory" in tex
        assert "StoreA" in tex or "Store" in tex

    def test_color_definitions(self, beer_game_graph):
        tex = render_graph(beer_game_graph, backend="tikz")
        assert r"\definecolor" in tex

    def test_node_drawing(self, consumer_graph):
        tex = render_graph(consumer_graph, backend="tikz")
        assert r"\node" in tex
        assert r"\draw" in tex


# ── Graphviz Backend Tests ──────────────────────────────────────────


class TestGraphvizBackend:
    def test_basic_render(self, beer_game_graph):
        graphviz = pytest.importorskip("graphviz")

        source = render_graph(beer_game_graph, backend="graphviz", theme="kfupm")
        assert source is not None
        assert "digraph" in source.source

    def test_all_themes(self, consumer_graph):
        graphviz = pytest.importorskip("graphviz")

        for theme_name in ["kfupm", "ieee", "presentation", "minimal"]:
            source = render_graph(
                consumer_graph, backend="graphviz", theme=theme_name
            )
            assert "digraph" in source.source

    def test_with_title(self, beer_game_graph):
        graphviz = pytest.importorskip("graphviz")

        source = render_graph(
            beer_game_graph, backend="graphviz", title="Beer Game"
        )
        assert "Beer Game" in source.source

    def test_with_sim_result(self, beer_game_graph, fake_result):
        graphviz = pytest.importorskip("graphviz")

        source = render_graph(
            beer_game_graph, backend="graphviz", sim_result=fake_result
        )
        assert "BW=" in source.source

    def test_tree_topology(self, tree_graph):
        graphviz = pytest.importorskip("graphviz")

        source = render_graph(tree_graph, backend="graphviz")
        assert "Factory" in source.source


# ── Unified API Tests ───────────────────────────────────────────────


class TestRenderGraphAPI:
    def test_unknown_backend_raises(self, consumer_graph):
        with pytest.raises(ValueError, match="Unknown backend"):
            render_graph(consumer_graph, backend="unknown")

    def test_string_theme_resolution(self, consumer_graph):
        import matplotlib.pyplot as plt

        fig = render_graph(consumer_graph, theme="ieee")
        assert fig is not None
        plt.close(fig)

    def test_theme_object(self, consumer_graph):
        import matplotlib.pyplot as plt

        theme = get_theme("minimal")
        fig = render_graph(consumer_graph, theme=theme)
        assert fig is not None
        plt.close(fig)


class TestRenderFromJson:
    def test_render_from_json(self, tmp_path, beer_game_graph):
        import matplotlib.pyplot as plt

        from deepbullwhip.render.api import render_from_json
        from deepbullwhip.schema.io import save_json

        path = str(tmp_path / "test.json")
        save_json(beer_game_graph, path, metadata={"name": "Test"})

        fig = render_from_json(path, backend="matplotlib", theme="kfupm")
        assert fig is not None
        plt.close(fig)

    def test_render_from_json_tikz(self, tmp_path, consumer_graph):
        from deepbullwhip.render.api import render_from_json
        from deepbullwhip.schema.io import save_json

        path = str(tmp_path / "consumer.json")
        save_json(consumer_graph, path)

        tex = render_from_json(path, backend="tikz", theme="ieee")
        assert r"\begin{tikzpicture}" in tex

    def test_render_from_json_with_hints(self, tmp_path, consumer_graph):
        import matplotlib.pyplot as plt

        from deepbullwhip.render.api import render_from_json
        from deepbullwhip.schema.definition import NodeLayoutHint
        from deepbullwhip.schema.io import save_json

        hints = {"Retailer": NodeLayoutHint(tier=0, position=(0, 0))}
        path = str(tmp_path / "hints.json")
        save_json(consumer_graph, path, layout_hints=hints)

        fig = render_from_json(path, backend="matplotlib")
        assert fig is not None
        plt.close(fig)
