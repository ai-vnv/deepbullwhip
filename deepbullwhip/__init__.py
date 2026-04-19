"""DeepBullwhip: Multi-tier supply chain bullwhip effect simulator.

A modular, extensible framework for simulating serial and network
supply chains and analyzing the bullwhip effect through various
demand patterns, ordering policies, and cost functions.

v0.3.0 adds:

- **Network topologies**: :class:`SupplyChainGraph` and :class:`EdgeConfig`
  for arbitrary DAG supply chains (trees, convergent/divergent networks).
- **NetworkX integration**: bidirectional graph conversion and analysis
  (requires ``pip install deepbullwhip[network]``).
- **Graphviz visualization**: publication-quality network rendering
  (requires ``pip install deepbullwhip[viz]``).
- **Pyomo optimization**: inventory optimization, policy tuning,
  and network design (requires ``pip install deepbullwhip[optimize]``).

Install all optional dependencies with::

    pip install deepbullwhip[all]
"""

# Core (v0.1.0 — backward compatible)
from deepbullwhip.chain.config import EchelonConfig, default_semiconductor_config
from deepbullwhip.chain.echelon import SupplyChainEchelon
from deepbullwhip.chain.serial import SerialSupplyChain
from deepbullwhip.chain.vectorized import BatchSimulationResult, VectorizedSupplyChain
from deepbullwhip.cost.newsvendor import NewsvendorCost
from deepbullwhip.demand.semiconductor import SemiconductorDemandGenerator
from deepbullwhip.policy.order_up_to import OrderUpToPolicy

# Registry (v0.2.0)
from deepbullwhip.registry import get, get_class, list_registered, register

# New demand generators (v0.2.0)
from deepbullwhip.demand.arma import ARMADemandGenerator
from deepbullwhip.demand.beer_game import BeerGameDemandGenerator
from deepbullwhip.demand.replay import ReplayDemandGenerator

# New policies (v0.2.0)
from deepbullwhip.policy.constant_order import ConstantOrderPolicy
from deepbullwhip.policy.proportional_out import ProportionalOUTPolicy
from deepbullwhip.policy.smoothing_out import SmoothingOUTPolicy

# New cost function (v0.2.0)
from deepbullwhip.cost.perishable import PerishableCost

# Forecasters (v0.2.0)
from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.forecast.exponential_smoothing import ExponentialSmoothingForecaster
from deepbullwhip.forecast.moving_average import MovingAverageForecaster
from deepbullwhip.forecast.naive import NaiveForecaster

# Benchmark (v0.2.0)
from deepbullwhip.benchmark.runner import BenchmarkRunner

# Network topologies (v0.3.0 — always available, no extra deps)
from deepbullwhip.chain.graph import EdgeConfig, SupplyChainGraph, from_serial
from deepbullwhip.chain.network_sim import NetworkSupplyChain
from deepbullwhip._types import NetworkSimulationResult

# NetworkX integration (v0.3.0 — functions use lazy imports internally)
from deepbullwhip.network.convert import from_networkx, serial_to_networkx, to_networkx

# Graphviz visualization (v0.3.0 — functions use lazy imports internally)
from deepbullwhip.diagnostics.graphviz_viz import (
    render_network,
    render_simulation_snapshot,
    save_figure,
)

# Schema serialization (v0.3.0)
from deepbullwhip.schema.io import (
    from_json,
    load_json,
    save_json,
    to_json,
)

# Multi-backend renderer (v0.3.0)
from deepbullwhip.render.api import render_from_json, render_graph
from deepbullwhip.render.theme import Theme, get_theme, list_themes, register_theme

__version__ = "0.5.0"

__all__ = [
    # Core (v0.1.0)
    "SemiconductorDemandGenerator",
    "OrderUpToPolicy",
    "NewsvendorCost",
    "SerialSupplyChain",
    "SupplyChainEchelon",
    "EchelonConfig",
    "default_semiconductor_config",
    "VectorizedSupplyChain",
    "BatchSimulationResult",
    # Registry (v0.2.0)
    "register",
    "get",
    "get_class",
    "list_registered",
    # New demand generators (v0.2.0)
    "BeerGameDemandGenerator",
    "ARMADemandGenerator",
    "ReplayDemandGenerator",
    # New policies (v0.2.0)
    "ProportionalOUTPolicy",
    "ConstantOrderPolicy",
    "SmoothingOUTPolicy",
    # New cost (v0.2.0)
    "PerishableCost",
    # Forecasters (v0.2.0)
    "Forecaster",
    "NaiveForecaster",
    "MovingAverageForecaster",
    "ExponentialSmoothingForecaster",
    # Benchmark (v0.2.0)
    "BenchmarkRunner",
    # Network topologies (v0.3.0)
    "SupplyChainGraph",
    "EdgeConfig",
    "from_serial",
    "NetworkSupplyChain",
    "NetworkSimulationResult",
    # NetworkX integration (v0.3.0)
    "to_networkx",
    "from_networkx",
    "serial_to_networkx",
    # Graphviz visualization (v0.3.0)
    "render_network",
    "render_simulation_snapshot",
    "save_figure",
    # Schema serialization (v0.3.0)
    "to_json",
    "from_json",
    "save_json",
    "load_json",
    # Multi-backend renderer (v0.3.0)
    "render_graph",
    "render_from_json",
    "Theme",
    "get_theme",
    "list_themes",
    "register_theme",
]
