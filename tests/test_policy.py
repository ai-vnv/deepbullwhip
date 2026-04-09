import numpy as np
import pytest
from scipy import stats

from deepbullwhip.policy.base import OrderingPolicy
from deepbullwhip.policy.order_up_to import OrderUpToPolicy


class TestOrderUpToPolicy:
    def test_basic_order(self):
        policy = OrderUpToPolicy(lead_time=2, service_level=0.95)
        z = stats.norm.ppf(0.95)
        fm, fs = 10.0, 2.0
        ip = 20.0
        S = 3 * fm + z * fs * np.sqrt(3)
        expected = max(0.0, S - ip)
        assert policy.compute_order(ip, fm, fs) == pytest.approx(expected)

    def test_order_nonnegative(self):
        policy = OrderUpToPolicy(lead_time=1, service_level=0.5)
        # IP much higher than S
        order = policy.compute_order(1000.0, 1.0, 0.1)
        assert order == 0.0

    def test_safety_stock_increases_with_service_level(self):
        p_low = OrderUpToPolicy(lead_time=4, service_level=0.80)
        p_high = OrderUpToPolicy(lead_time=4, service_level=0.99)
        o_low = p_low.compute_order(0.0, 10.0, 2.0)
        o_high = p_high.compute_order(0.0, 10.0, 2.0)
        assert o_high > o_low

    def test_lead_time_effect(self):
        p_short = OrderUpToPolicy(lead_time=2, service_level=0.95)
        p_long = OrderUpToPolicy(lead_time=10, service_level=0.95)
        o_short = p_short.compute_order(0.0, 10.0, 2.0)
        o_long = p_long.compute_order(0.0, 10.0, 2.0)
        assert o_long > o_short

    def test_zero_forecast_std(self):
        policy = OrderUpToPolicy(lead_time=3, service_level=0.95)
        order = policy.compute_order(0.0, 10.0, 0.0)
        # S = (3+1)*10 + z*0 = 40
        assert order == pytest.approx(40.0)

    def test_abc_interface(self):
        policy = OrderUpToPolicy(lead_time=2)
        assert isinstance(policy, OrderingPolicy)
