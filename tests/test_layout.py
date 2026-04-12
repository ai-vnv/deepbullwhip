"""Tests for the layout engine."""


from deepbullwhip.chain.config import (
    EchelonConfig,
    beer_game_config,
    consumer_2tier_config,
)
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph, from_serial
from deepbullwhip.render.layout import (
    compute_figure_size,
    compute_positions,
    compute_tiers,
)
from deepbullwhip.render.theme import get_theme
from deepbullwhip.schema.definition import LayoutDefaults, NodeLayoutHint


class TestComputeTiers:
    def test_serial_chain(self):
        graph = from_serial(beer_game_config())
        tiers = compute_tiers(graph)
        assert tiers["Factory"] == 0  # source
        assert tiers["Retailer"] == 3  # demand-facing

    def test_2tier_chain(self):
        graph = from_serial(consumer_2tier_config())
        tiers = compute_tiers(graph)
        assert tiers["Manufacturer"] == 0
        assert tiers["Retailer"] == 1

    def test_tree_topology(self):
        graph = SupplyChainGraph(
            nodes={
                "Factory": EchelonConfig("Factory", 4, 0.1, 0.4),
                "WH_A": EchelonConfig("WH_A", 2, 0.15, 0.5),
                "WH_B": EchelonConfig("WH_B", 2, 0.15, 0.5),
                "Retail": EchelonConfig("Retail", 1, 0.2, 0.6),
            },
            edges={
                ("Factory", "WH_A"): EdgeConfig(),
                ("Factory", "WH_B"): EdgeConfig(),
                ("WH_A", "Retail"): EdgeConfig(),
            },
        )
        tiers = compute_tiers(graph)
        assert tiers["Factory"] == 0
        assert tiers["WH_A"] == 1
        assert tiers["WH_B"] == 1
        assert tiers["Retail"] == 2

    def test_single_node(self):
        graph = SupplyChainGraph(
            nodes={"Solo": EchelonConfig("Solo", 1, 0.1, 0.5)},
            edges={},
        )
        tiers = compute_tiers(graph)
        assert tiers["Solo"] == 0


class TestComputePositions:
    def test_serial_chain_tb(self):
        graph = from_serial(beer_game_config())
        pos = compute_positions(graph)
        assert len(pos) == 4
        # In TB mode, Factory (tier 0) has highest y
        assert pos["Factory"][1] > pos["Retailer"][1]

    def test_serial_chain_lr(self):
        graph = from_serial(beer_game_config())
        pos = compute_positions(graph, defaults=LayoutDefaults(orientation="LR"))
        # In LR mode, Factory (tier 0) has smallest x
        assert pos["Factory"][0] < pos["Retailer"][0]

    def test_layout_hints_override(self):
        graph = from_serial(consumer_2tier_config())
        hints = {
            "Retailer": NodeLayoutHint(position=(5.0, 10.0)),
        }
        pos = compute_positions(graph, layout_hints=hints)
        assert pos["Retailer"] == (5.0, 10.0)

    def test_tier_hint_override(self):
        graph = from_serial(beer_game_config())
        # Override Factory from tier 0 to tier 5 -- moves it farther from Retailer
        hints = {
            "Factory": NodeLayoutHint(tier=10),
        }
        pos = compute_positions(graph, layout_hints=hints)
        default_pos = compute_positions(graph)
        # Factory's y position should change because tier range expanded
        assert pos["Factory"][1] != default_pos["Factory"][1]

    def test_tree_positions(self):
        graph = SupplyChainGraph(
            nodes={
                "F": EchelonConfig("F", 4, 0.1, 0.4),
                "A": EchelonConfig("A", 1, 0.2, 0.6),
                "B": EchelonConfig("B", 1, 0.2, 0.6),
            },
            edges={
                ("F", "A"): EdgeConfig(),
                ("F", "B"): EdgeConfig(),
            },
        )
        pos = compute_positions(graph)
        # A and B should be at the same tier (same y in TB)
        assert pos["A"][1] == pos["B"][1]
        # A and B should be at different x positions
        assert pos["A"][0] != pos["B"][0]

    def test_custom_spacing(self):
        graph = from_serial(consumer_2tier_config())
        pos_default = compute_positions(graph)
        pos_wide = compute_positions(
            graph, defaults=LayoutDefaults(tier_spacing=5.0)
        )
        # Wider spacing means greater y difference
        y_diff_default = abs(pos_default["Manufacturer"][1] - pos_default["Retailer"][1])
        y_diff_wide = abs(pos_wide["Manufacturer"][1] - pos_wide["Retailer"][1])
        assert y_diff_wide > y_diff_default


class TestComputeFigureSize:
    def test_returns_tuple(self):
        graph = from_serial(beer_game_config())
        pos = compute_positions(graph)
        theme = get_theme("kfupm")
        size = compute_figure_size(pos, theme)
        assert len(size) == 2
        assert size[0] > 0
        assert size[1] > 0

    def test_explicit_height(self):
        from deepbullwhip.render.theme import FigureStyle

        theme = get_theme("kfupm").override(figure=FigureStyle(width=5.0, height=3.0))
        size = compute_figure_size({}, theme)
        assert size == (5.0, 3.0)

    def test_empty_positions(self):
        theme = get_theme("kfupm")
        size = compute_figure_size({}, theme)
        assert size[0] == theme.figure.width
