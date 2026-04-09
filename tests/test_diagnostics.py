"""Tests for diagnostics plots and network visualizations.

Covers all plot functions and network diagram/map functions with
>95% coverage of the diagnostics subpackage.
"""

import matplotlib
import matplotlib.figure
import numpy as np
import pytest

matplotlib.use("Agg")

from deepbullwhip import (
    EchelonConfig,
    SemiconductorDemandGenerator,
    SerialSupplyChain,
)
from deepbullwhip.diagnostics.network import (
    NodeLocation,
    SupplyChainNetwork,
    kfupm_petrochemical_network,
    plot_network_diagram,
    plot_supply_chain_map,
)
from deepbullwhip.diagnostics.plots import (
    COLORS,
    _apply_style,
    _col_width,
    _echelon_color,
    plot_bullwhip_amplification,
    plot_cost_decomposition,
    plot_cost_timeseries,
    plot_demand_trajectory,
    plot_echelon_detail,
    plot_inventory_levels,
    plot_inventory_position,
    plot_order_quantities,
    plot_order_streams,
    plot_summary_dashboard,
)

import matplotlib.pyplot as plt


@pytest.fixture
def demand():
    gen = SemiconductorDemandGenerator()
    return gen.generate(T=52, seed=42)


@pytest.fixture
def chain_and_result(demand):
    chain = SerialSupplyChain()
    fm = np.full_like(demand, demand.mean())
    fs = np.full_like(demand, demand.std())
    result = chain.simulate(demand, fm, fs)
    return chain, result


@pytest.fixture
def sim_result(chain_and_result):
    _, result = chain_and_result
    return result


@pytest.fixture
def chain(chain_and_result):
    chain, _ = chain_and_result
    return chain


def _close(fig):
    plt.close(fig)


# ── Style helpers ────────────────────────────────────────────────────


class TestStyleHelpers:
    def test_apply_style(self):
        _apply_style()
        assert "serif" in plt.rcParams["font.family"]
        assert plt.rcParams["font.size"] == 8

    def test_col_width_single(self):
        assert _col_width("single") == pytest.approx(3.5)

    def test_col_width_double(self):
        assert _col_width("double") == pytest.approx(7.0)

    def test_echelon_color_wraps(self):
        c0 = _echelon_color(0)
        c4 = _echelon_color(4)
        assert c0 == c4  # wraps around

    def test_colors_dict_has_required_keys(self):
        for key in ["demand", "E1", "E2", "E3", "E4", "holding", "backorder", "grid"]:
            assert key in COLORS


# ── Demand trajectory ────────────────────────────────────────────────


class TestPlotDemandTrajectory:
    def test_returns_figure(self, demand):
        fig = plot_demand_trajectory(demand)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2
        _close(fig)

    def test_single_column(self, demand):
        fig = plot_demand_trajectory(demand, width="single")
        w, h = fig.get_size_inches()
        assert w == pytest.approx(3.5)
        _close(fig)

    def test_no_shock(self, demand):
        fig = plot_demand_trajectory(demand, shock_period=None)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_shock_beyond_T(self, demand):
        fig = plot_demand_trajectory(demand, shock_period=9999)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)


# ── Order quantities ─────────────────────────────────────────────────


class TestPlotOrderQuantities:
    def test_returns_figure(self, demand, sim_result):
        fig = plot_order_quantities(demand, sim_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        # K+1 axes (1 demand + K echelons)
        assert len(fig.axes) == len(sim_result.echelon_results) + 1
        _close(fig)

    def test_single_column(self, demand, sim_result):
        fig = plot_order_quantities(demand, sim_result, width="single")
        w, _ = fig.get_size_inches()
        assert w == pytest.approx(3.5)
        _close(fig)


# ── Inventory levels ─────────────────────────────────────────────────


class TestPlotInventoryLevels:
    def test_returns_figure(self, sim_result):
        fig = plot_inventory_levels(sim_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == len(sim_result.echelon_results)
        _close(fig)

    def test_single_echelon(self):
        configs = [EchelonConfig("A", lead_time=1, holding_cost=0.1, backorder_cost=0.5)]
        chain = SerialSupplyChain.from_config(configs)
        demand = np.full(20, 10.0)
        fm = np.full(20, 10.0)
        fs = np.full(20, 1.0)
        result = chain.simulate(demand, fm, fs)
        fig = plot_inventory_levels(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)


# ── Inventory position ───────────────────────────────────────────────


class TestPlotInventoryPosition:
    def test_returns_figure(self, demand, sim_result, chain):
        fig = plot_inventory_position(demand, sim_result, chain)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_single_echelon(self):
        configs = [EchelonConfig("A", lead_time=2, holding_cost=0.1, backorder_cost=0.5)]
        chain = SerialSupplyChain.from_config(configs)
        demand = np.full(20, 10.0)
        fm = np.full(20, 10.0)
        fs = np.full(20, 1.0)
        result = chain.simulate(demand, fm, fs)
        fig = plot_inventory_position(demand, result, chain)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)


# ── Order streams ────────────────────────────────────────────────────


class TestPlotOrderStreams:
    def test_default_echelons(self, demand, sim_result):
        fig = plot_order_streams(demand, sim_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_specific_echelons(self, demand, sim_result):
        fig = plot_order_streams(demand, sim_result, echelon_indices=[0, 3])
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)


# ── Cost time series ─────────────────────────────────────────────────


class TestPlotCostTimeseries:
    def test_returns_figure(self, sim_result):
        fig = plot_cost_timeseries(sim_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == len(sim_result.echelon_results)
        _close(fig)

    def test_single_echelon(self):
        configs = [EchelonConfig("A", lead_time=1, holding_cost=0.1, backorder_cost=0.5)]
        chain = SerialSupplyChain.from_config(configs)
        demand = np.full(20, 10.0)
        result = chain.simulate(demand, np.full(20, 10.0), np.full(20, 1.0))
        fig = plot_cost_timeseries(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)


# ── Cost decomposition ───────────────────────────────────────────────


class TestPlotCostDecomposition:
    def test_single_model(self, sim_result):
        fig = plot_cost_decomposition({"model_A": sim_result})
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_multiple_models(self, sim_result):
        fig = plot_cost_decomposition({"A": sim_result, "B": sim_result})
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)


# ── Bullwhip amplification ───────────────────────────────────────────


class TestPlotBullwhipAmplification:
    def test_returns_figure(self, sim_result):
        fig = plot_bullwhip_amplification({"default": sim_result})
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_with_labels(self, sim_result):
        fig = plot_bullwhip_amplification(
            {"default": sim_result},
            echelon_labels=["D", "O", "F", "S"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_single_column(self, sim_result):
        fig = plot_bullwhip_amplification({"x": sim_result}, width="single")
        w, _ = fig.get_size_inches()
        assert w == pytest.approx(3.5)
        _close(fig)


# ── Summary dashboard ────────────────────────────────────────────────


class TestPlotSummaryDashboard:
    def test_returns_figure(self, demand, sim_result):
        fig = plot_summary_dashboard(demand, sim_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        # 2x2 grid = 4 axes + 1 twin = 5
        assert len(fig.axes) >= 4
        _close(fig)


# ── Echelon detail ───────────────────────────────────────────────────


class TestPlotEchelonDetail:
    def test_first_echelon(self, demand, sim_result):
        fig = plot_echelon_detail(demand, sim_result, echelon_index=0)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3
        _close(fig)

    def test_last_echelon(self, demand, sim_result):
        K = len(sim_result.echelon_results)
        fig = plot_echelon_detail(demand, sim_result, echelon_index=K - 1)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_single_column(self, demand, sim_result):
        fig = plot_echelon_detail(demand, sim_result, width="single")
        w, _ = fig.get_size_inches()
        assert w == pytest.approx(3.5)
        _close(fig)


# ═══════════════════════════════════════════════════════════════════
# Network diagram and map tests
# ═══════════════════════════════════════════════════════════════════


class TestNodeLocation:
    def test_basic_creation(self):
        node = NodeLocation("KFUPM", lat=26.3, lon=50.1)
        assert node.name == "KFUPM"
        assert node.lat == 26.3
        assert node.role == ""

    def test_with_role(self):
        node = NodeLocation("KFUPM", lat=26.3, lon=50.1, role="Distributor", details="Dhahran")
        assert node.role == "Distributor"
        assert node.details == "Dhahran"


class TestSupplyChainNetwork:
    def test_auto_edges(self):
        nodes = [NodeLocation(f"N{i}", lat=i, lon=i) for i in range(3)]
        net = SupplyChainNetwork(nodes=nodes)
        assert net.edges == [(1, 0), (2, 1)]

    def test_explicit_edges(self):
        nodes = [NodeLocation(f"N{i}", lat=i, lon=i) for i in range(3)]
        net = SupplyChainNetwork(nodes=nodes, edges=[(0, 2), (2, 1)])
        assert net.edges == [(0, 2), (2, 1)]

    def test_single_node_no_edges(self):
        nodes = [NodeLocation("A", lat=0, lon=0)]
        net = SupplyChainNetwork(nodes=nodes)
        assert net.edges == []


class TestKfupmNetwork:
    def test_returns_network(self):
        net = kfupm_petrochemical_network()
        assert isinstance(net, SupplyChainNetwork)
        assert len(net.nodes) == 4
        assert len(net.edges) == 3

    def test_node_names(self):
        net = kfupm_petrochemical_network()
        names = [n.name for n in net.nodes]
        assert "KFUPM / Distributor" in names

    def test_has_coordinates(self):
        net = kfupm_petrochemical_network()
        for node in net.nodes:
            assert 15 < node.lat < 35  # Saudi Arabia latitude range
            assert 34 < node.lon < 56  # Saudi Arabia longitude range


class TestPlotNetworkDiagram:
    def test_returns_figure(self):
        net = kfupm_petrochemical_network()
        fig = plot_network_diagram(net)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_with_sim_result(self, sim_result):
        net = kfupm_petrochemical_network()
        fig = plot_network_diagram(net, sim_result=sim_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_vertical_orientation(self):
        net = kfupm_petrochemical_network()
        fig = plot_network_diagram(net, orientation="vertical")
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_single_column(self):
        net = kfupm_petrochemical_network()
        fig = plot_network_diagram(net, width="single")
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_two_node_network(self):
        nodes = [
            NodeLocation("A", lat=0, lon=0, role="Buyer"),
            NodeLocation("B", lat=1, lon=1, role="Seller"),
        ]
        net = SupplyChainNetwork(nodes=nodes)
        fig = plot_network_diagram(net)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_single_node(self):
        nodes = [NodeLocation("A", lat=0, lon=0, role="Solo")]
        net = SupplyChainNetwork(nodes=nodes)
        fig = plot_network_diagram(net)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)


class TestPlotSupplyChainMap:
    def test_returns_figure(self):
        net = kfupm_petrochemical_network()
        fig = plot_supply_chain_map(net)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_with_sim_result(self, sim_result):
        net = kfupm_petrochemical_network()
        fig = plot_supply_chain_map(net, sim_result=sim_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_custom_bounds(self):
        net = kfupm_petrochemical_network()
        fig = plot_supply_chain_map(net, map_bounds=(20, 30, 35, 55))
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_no_country_outline(self):
        net = kfupm_petrochemical_network()
        fig = plot_supply_chain_map(net, show_country_outline=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_single_column(self):
        net = kfupm_petrochemical_network()
        fig = plot_supply_chain_map(net, width="single")
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)

    def test_two_node_map(self):
        nodes = [
            NodeLocation("A", lat=26.0, lon=50.0),
            NodeLocation("B", lat=24.0, lon=47.0),
        ]
        net = SupplyChainNetwork(nodes=nodes)
        fig = plot_supply_chain_map(net)
        assert isinstance(fig, matplotlib.figure.Figure)
        _close(fig)
