import numpy as np
import pytest

from deepbullwhip import (
    NewsvendorCost,
    OrderUpToPolicy,
    SemiconductorDemandGenerator,
    SerialSupplyChain,
    SupplyChainEchelon,
)


@pytest.fixture
def demand_generator():
    return SemiconductorDemandGenerator()


@pytest.fixture
def demand_series(demand_generator):
    return demand_generator.generate(T=156, seed=42)


@pytest.fixture
def default_chain():
    return SerialSupplyChain()


@pytest.fixture
def simple_echelon():
    """Single echelon with known parameters for unit testing."""
    policy = OrderUpToPolicy(lead_time=2, service_level=0.95)
    cost_fn = NewsvendorCost(holding_cost=0.15, backorder_cost=0.60)
    return SupplyChainEchelon(
        "test", lead_time=2, policy=policy, cost_fn=cost_fn, initial_inventory=50.0
    )


@pytest.fixture
def constant_demand():
    """Constant demand of 10.0 for 100 periods."""
    return np.full(100, 10.0)
