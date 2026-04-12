"""Tests for JSON schema serialization and deserialization."""

import json
import math

import pytest

from deepbullwhip.chain.config import (
    EchelonConfig,
    beer_game_config,
    consumer_2tier_config,
    default_semiconductor_config,
)
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph, from_serial
from deepbullwhip.schema.definition import (
    SCHEMA_VERSION,
    LayoutDefaults,
    NetworkMetadata,
    NodeLayoutHint,
)
from deepbullwhip.schema.io import (
    from_dict,
    from_json,
    load_json,
    load_json_full,
    save_json,
    to_dict,
    to_json,
)


class TestSchemaVersion:
    def test_version_is_string(self):
        assert isinstance(SCHEMA_VERSION, str)
        assert SCHEMA_VERSION == "1.0"


class TestNodeLayoutHint:
    def test_defaults(self):
        hint = NodeLayoutHint()
        assert hint.tier is None
        assert hint.role == ""
        assert hint.position is None
        assert hint.label is None

    def test_custom_values(self):
        hint = NodeLayoutHint(tier=2, role="distributor", position=(1.0, 2.0), label="WH")
        assert hint.tier == 2
        assert hint.role == "distributor"
        assert hint.position == (1.0, 2.0)
        assert hint.label == "WH"


class TestLayoutDefaults:
    def test_defaults(self):
        d = LayoutDefaults()
        assert d.orientation == "TB"
        assert d.tier_spacing == 3.0
        assert d.node_spacing == 3.0
        assert d.auto_position is True


class TestNetworkMetadata:
    def test_defaults(self):
        m = NetworkMetadata()
        assert m.name == ""
        assert m.tags == []


class TestToDict:
    def test_beer_game(self):
        graph = from_serial(beer_game_config())
        d = to_dict(graph)
        assert d["version"] == "1.0"
        assert len(d["nodes"]) == 4
        assert len(d["edges"]) == 3

    def test_node_fields(self):
        graph = from_serial(consumer_2tier_config())
        d = to_dict(graph)
        node = d["nodes"][0]
        assert "id" in node
        assert "config" in node
        assert "lead_time" in node["config"]
        assert "holding_cost" in node["config"]
        assert "backorder_cost" in node["config"]

    def test_edge_fields(self):
        graph = from_serial(beer_game_config())
        d = to_dict(graph)
        edge = d["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "config" in edge

    def test_infinity_serialized_as_null(self):
        graph = SupplyChainGraph(
            nodes={"A": EchelonConfig("A", 1, 0.1, 0.5)},
            edges={},
        )
        to_dict(graph)
        # No edges, but test the serializer handles inf in EdgeConfig
        graph2 = SupplyChainGraph(
            nodes={
                "A": EchelonConfig("A", 1, 0.1, 0.5),
                "B": EchelonConfig("B", 1, 0.1, 0.5),
            },
            edges={("A", "B"): EdgeConfig(capacity=float("inf"))},
        )
        d2 = to_dict(graph2)
        assert d2["edges"][0]["config"]["capacity"] is None

    def test_with_metadata(self):
        graph = from_serial(consumer_2tier_config())
        d = to_dict(graph, metadata={"name": "Test", "tags": ["serial"]})
        assert d["metadata"]["name"] == "Test"
        assert d["metadata"]["tags"] == ["serial"]

    def test_with_metadata_object(self):
        graph = from_serial(consumer_2tier_config())
        meta = NetworkMetadata(name="Test", author="User")
        d = to_dict(graph, metadata=meta)
        assert d["metadata"]["name"] == "Test"
        assert d["metadata"]["author"] == "User"

    def test_with_layout_hints(self):
        graph = from_serial(consumer_2tier_config())
        hints = {
            "Retailer": NodeLayoutHint(tier=0, role="retailer", position=(0, 0)),
        }
        d = to_dict(graph, layout_hints=hints)
        retailer = next(n for n in d["nodes"] if n["id"] == "Retailer")
        assert "layout" in retailer
        assert retailer["layout"]["tier"] == 0
        assert retailer["layout"]["position"] == [0, 0]

    def test_with_layout_defaults(self):
        graph = from_serial(consumer_2tier_config())
        defaults = LayoutDefaults(orientation="LR", tier_spacing=3.0)
        d = to_dict(graph, layout_defaults=defaults)
        assert d["layout_defaults"]["orientation"] == "LR"
        assert d["layout_defaults"]["tier_spacing"] == 3.0


class TestFromDict:
    def test_beer_game_round_trip(self):
        original = from_serial(beer_game_config())
        d = to_dict(original)
        restored = from_dict(d)
        assert set(restored.nodes.keys()) == set(original.nodes.keys())
        assert set(restored.edges.keys()) == set(original.edges.keys())

    def test_config_values_preserved(self):
        original = from_serial(consumer_2tier_config())
        d = to_dict(original)
        restored = from_dict(d)
        for name in original.nodes:
            orig_cfg = original.nodes[name]
            rest_cfg = restored.nodes[name]
            assert rest_cfg.lead_time == orig_cfg.lead_time
            assert rest_cfg.holding_cost == pytest.approx(orig_cfg.holding_cost)
            assert rest_cfg.backorder_cost == pytest.approx(orig_cfg.backorder_cost)

    def test_null_capacity_becomes_inf(self):
        d = {
            "nodes": [
                {"id": "A", "config": {"lead_time": 1, "holding_cost": 0.1, "backorder_cost": 0.5}},
                {"id": "B", "config": {"lead_time": 1, "holding_cost": 0.1, "backorder_cost": 0.5}},
            ],
            "edges": [
                {"source": "A", "target": "B", "config": {"lead_time": 1, "capacity": None}},
            ],
        }
        graph = from_dict(d)
        assert math.isinf(graph.edges[("A", "B")].capacity)

    def test_missing_config_fields_use_defaults(self):
        d = {
            "nodes": [{"id": "X", "config": {"lead_time": 3, "holding_cost": 0.2, "backorder_cost": 0.8}}],
            "edges": [],
        }
        graph = from_dict(d)
        cfg = graph.nodes["X"]
        assert cfg.depreciation_rate == 0.0  # default
        assert cfg.service_level == 0.95  # default
        assert cfg.initial_inventory == 50.0  # default


class TestJsonRoundTrip:
    def test_beer_game(self):
        original = from_serial(beer_game_config())
        json_str = to_json(original)
        restored = from_json(json_str)
        assert set(restored.nodes.keys()) == set(original.nodes.keys())

    def test_semiconductor(self):
        original = from_serial(default_semiconductor_config())
        json_str = to_json(original)
        restored = from_json(json_str)
        assert len(restored.nodes) == 4
        assert len(restored.edges) == 3

    def test_json_is_valid(self):
        graph = from_serial(beer_game_config())
        json_str = to_json(graph)
        parsed = json.loads(json_str)
        assert parsed["version"] == "1.0"

    def test_with_all_options(self):
        graph = from_serial(beer_game_config())
        meta = NetworkMetadata(name="Full Test", tags=["test"])
        hints = {"Factory": NodeLayoutHint(tier=3, role="manufacturer")}
        defaults = LayoutDefaults(orientation="LR")
        json_str = to_json(graph, metadata=meta, layout_hints=hints, layout_defaults=defaults)
        parsed = json.loads(json_str)
        assert parsed["metadata"]["name"] == "Full Test"
        assert "layout_defaults" in parsed


class TestFileIO:
    def test_save_and_load(self, tmp_path):
        graph = from_serial(beer_game_config())
        path = str(tmp_path / "test.json")
        save_json(graph, path, metadata={"name": "Test"})
        restored = load_json(path)
        assert set(restored.nodes.keys()) == set(graph.nodes.keys())

    def test_load_full(self, tmp_path):
        graph = from_serial(beer_game_config())
        hints = {"Factory": NodeLayoutHint(tier=3, role="mfg", position=(0, 3))}
        path = str(tmp_path / "full.json")
        save_json(graph, path, metadata={"name": "Full"}, layout_hints=hints)

        restored_graph, metadata, layout_hints = load_json_full(path)
        assert metadata.name == "Full"
        assert "Factory" in layout_hints
        assert layout_hints["Factory"].tier == 3
        assert layout_hints["Factory"].position == (0, 3)
