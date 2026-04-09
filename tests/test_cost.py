import pytest

from deepbullwhip.cost.base import CostFunction
from deepbullwhip.cost.newsvendor import NewsvendorCost


class TestNewsvendorCost:
    def test_holding_cost_positive_inventory(self):
        cost_fn = NewsvendorCost(holding_cost=0.15, backorder_cost=0.60)
        assert cost_fn.compute(10.0) == pytest.approx(1.5)

    def test_backorder_cost_negative_inventory(self):
        cost_fn = NewsvendorCost(holding_cost=0.15, backorder_cost=0.60)
        assert cost_fn.compute(-5.0) == pytest.approx(3.0)

    def test_zero_inventory(self):
        cost_fn = NewsvendorCost(holding_cost=0.15, backorder_cost=0.60)
        assert cost_fn.compute(0.0) == pytest.approx(0.0)

    def test_large_positive_inventory(self):
        cost_fn = NewsvendorCost(holding_cost=0.10, backorder_cost=0.50)
        assert cost_fn.compute(1000.0) == pytest.approx(100.0)

    def test_abc_interface(self):
        cost_fn = NewsvendorCost(holding_cost=0.1, backorder_cost=0.5)
        assert isinstance(cost_fn, CostFunction)
