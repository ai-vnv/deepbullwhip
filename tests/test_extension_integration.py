"""Integration tests for ``deepbullwhip.ext`` (learning components)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import deepbullwhip.ext  # noqa: F401  # registers extension components

from deepbullwhip.benchmark.runner import BenchmarkRunner
from deepbullwhip.registry import get, get_class, list_registered


def test_all_extension_components_registered():
    reg = list_registered()
    policies = set(reg["policy"])
    forecasters = set(reg["forecaster"])
    metrics = set(reg["metric"])

    assert {"dqn_beer_game", "recurrent_ppo", "dcl", "e2e_newsvendor"} <= policies
    assert {"nbeats", "tft", "lightgbm_quantile", "lstm_multistep"} <= forecasters
    assert {
        "RFU",
        "OSR",
        "PeakBWR",
        "ExpectedShortfall",
        "InventoryTurnover",
        "DampingRatio",
    } <= metrics


@pytest.mark.parametrize(
    "policy_name",
    ["dqn_beer_game", "recurrent_ppo", "dcl", "e2e_newsvendor"],
)
def test_policy_fallback_produces_valid_orders(policy_name):
    cls = get_class("policy", policy_name)
    policy = cls(lead_time=4, service_level=0.9)
    for ip, fm, fs in [(10.0, 12.0, 2.0), (-5.0, 15.0, 3.0), (50.0, 8.0, 1.0)]:
        q = policy.compute_order(ip, fm, fs)
        assert isinstance(q, float)
        assert math.isfinite(q)
        assert q >= 0.0


@pytest.mark.parametrize(
    "fc_name",
    ["nbeats", "tft", "lightgbm_quantile", "lstm_multistep"],
)
def test_forecaster_fallback_produces_valid_outputs(fc_name):
    rng = np.random.default_rng(0)
    hist = 100 + 5 * np.sin(np.linspace(0, 8 * np.pi, 80)) + rng.normal(0, 3, 80)
    fc = get("forecaster", fc_name)
    mean, std = fc.forecast(hist)
    assert math.isfinite(mean) and math.isfinite(std)
    assert mean >= 0.0
    assert std >= 0.0

    fm, fs = fc.generate_forecasts(hist)
    assert fm.shape == hist.shape
    assert fs.shape == hist.shape
    assert np.all(np.isfinite(fm)) and np.all(np.isfinite(fs))
    assert np.all(fm >= 0.0) and np.all(fs >= 0.0)


def _mini_sim_result():
    from deepbullwhip import SemiconductorDemandGenerator, SerialSupplyChain

    gen = SemiconductorDemandGenerator()
    demand = gen.generate(T=60, seed=0)
    chain = SerialSupplyChain()
    fm = np.full_like(demand, demand.mean())
    fs = np.full_like(demand, demand.std())
    result = chain.simulate(demand, fm, fs)
    return result, demand


@pytest.mark.parametrize(
    "metric_name",
    [
        "RFU",
        "OSR",
        "PeakBWR",
        "ExpectedShortfall",
        "InventoryTurnover",
        "DampingRatio",
    ],
)
def test_new_metrics_compute(metric_name):
    result, demand = _mini_sim_result()
    cls = get_class("metric", metric_name)
    for k in range(len(result.echelon_results)):
        value = cls.compute(result, demand, echelon=k)
        assert isinstance(value, float)
        if not math.isnan(value):
            assert math.isfinite(value) or value == float("inf")


def test_full_benchmark_runner_end_to_end():
    runner = BenchmarkRunner(T=52, N=3, seed=42)

    df = runner.run(
        policies=[
            "order_up_to",
            "dcl",
            "recurrent_ppo",
            "e2e_newsvendor",
            ("dqn_beer_game", {"state_window": 4}),
        ],
        forecasters=[
            "naive",
            "moving_average",
            "nbeats",
            "tft",
            "lightgbm_quantile",
            "lstm_multistep",
        ],
        metrics=[
            "BWR",
            "CUM_BWR",
            "FILL_RATE",
            "TC",
            "NSAmp",
            "RFU",
            "OSR",
            "PeakBWR",
            "ExpectedShortfall",
            "InventoryTurnover",
            "DampingRatio",
        ],
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5 * 6 * 4 * 11
    assert set(df.columns) >= {"policy", "forecaster", "echelon", "metric", "value"}

    num_nan = df["value"].isna().sum()
    nan_rows = df[df["value"].isna()]
    assert set(nan_rows["metric"].unique()) <= {"DampingRatio"}
    assert num_nan < len(df) * 0.2


def test_leaderboard_aggregates_correctly():
    runner = BenchmarkRunner(T=52, N=2, seed=1)
    df = runner.run(
        policies=["order_up_to", "dcl"],
        forecasters=["naive", "nbeats"],
        metrics=["BWR", "CUM_BWR", "TC", "RFU", "OSR"],
    )
    last = df[df["echelon"] == "E4"]
    pivot = last.pivot_table(
        index=["policy", "forecaster"],
        columns="metric",
        values="value",
    )
    assert set(pivot.columns) == {"BWR", "CUM_BWR", "TC", "RFU", "OSR"}
    assert len(pivot) == 4
