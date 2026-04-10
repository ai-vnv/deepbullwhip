"""DeepBullwhip: Multi-tier supply chain bullwhip effect simulator."""

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

__version__ = "0.2.0"

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
]
