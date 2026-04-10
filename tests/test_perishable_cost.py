"""Tests for the PerishableCost function."""

import pytest

from deepbullwhip.cost.perishable import PerishableCost
from deepbullwhip.registry import list_registered


class TestPerishableCost:
    def test_positive_inventory_below_buffer(self):
        cost = PerishableCost(holding_cost=0.10, backorder_cost=0.50, gamma=0.05, buffer=50.0)
        # inventory=30 < buffer=50, no obsolescence
        c = cost.compute(30.0)
        assert c == pytest.approx(0.10 * 30.0)

    def test_positive_inventory_above_buffer(self):
        cost = PerishableCost(holding_cost=0.10, backorder_cost=0.50, gamma=0.05, buffer=50.0)
        # inventory=70, obsolescence on 20 units
        c = cost.compute(70.0)
        expected = 0.10 * 70.0 + 0.05 * 20.0
        assert c == pytest.approx(expected)

    def test_backorder(self):
        cost = PerishableCost(holding_cost=0.10, backorder_cost=0.50, gamma=0.05, buffer=50.0)
        c = cost.compute(-10.0)
        assert c == pytest.approx(0.50 * 10.0)

    def test_zero_inventory(self):
        cost = PerishableCost(holding_cost=0.10, backorder_cost=0.50)
        c = cost.compute(0.0)
        assert c == pytest.approx(0.0)

    def test_matches_newsvendor_below_buffer(self):
        """Below buffer, should behave like NewsvendorCost."""
        from deepbullwhip.cost.newsvendor import NewsvendorCost

        nv = NewsvendorCost(holding_cost=0.10, backorder_cost=0.50)
        per = PerishableCost(holding_cost=0.10, backorder_cost=0.50, gamma=0.05, buffer=100.0)

        for inv in [0.0, 10.0, 50.0, -5.0, -20.0]:
            assert per.compute(inv) == pytest.approx(nv.compute(inv))

    def test_registered(self):
        assert "perishable" in list_registered("cost")
