"""Tests for the BenchmarkRunner."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from deepbullwhip.benchmark.configs import PREDEFINED_CONFIGS
from deepbullwhip.benchmark.report import to_latex, to_markdown
from deepbullwhip.benchmark.runner import BenchmarkRunner
from deepbullwhip.chain.config import EchelonConfig


class TestBenchmarkRunner:
    def test_init_string_config(self):
        runner = BenchmarkRunner(chain_config="semiconductor_4tier", N=1)
        assert len(runner.configs) == 4

    def test_init_beer_game_config(self):
        runner = BenchmarkRunner(chain_config="beer_game", N=1)
        assert len(runner.configs) == 4

    def test_init_custom_config(self):
        configs = [
            EchelonConfig("E1", lead_time=2, holding_cost=0.1, backorder_cost=0.5),
        ]
        runner = BenchmarkRunner(chain_config=configs, N=1)
        assert len(runner.configs) == 1

    def test_init_unknown_config(self):
        with pytest.raises(KeyError, match="Unknown chain config"):
            BenchmarkRunner(chain_config="nonexistent")

    def test_run_returns_dataframe(self):
        runner = BenchmarkRunner(
            chain_config="consumer_2tier",
            demand="semiconductor_ar1",
            T=52,
            N=2,
            seed=42,
        )
        df = runner.run(policies=["order_up_to"])
        assert isinstance(df, pd.DataFrame)
        assert "policy" in df.columns
        assert "forecaster" in df.columns
        assert "echelon" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns

    def test_run_multiple_policies(self):
        runner = BenchmarkRunner(
            chain_config="consumer_2tier",
            demand="semiconductor_ar1",
            T=52,
            N=2,
            seed=42,
        )
        df = runner.run(
            policies=["order_up_to", ("proportional_out", {"alpha": 0.5})],
        )
        policies_in_result = df["policy"].unique()
        assert "order_up_to" in policies_in_result
        assert "proportional_out" in policies_in_result

    def test_run_multiple_forecasters(self):
        runner = BenchmarkRunner(
            chain_config="consumer_2tier",
            demand="semiconductor_ar1",
            T=52,
            N=2,
            seed=42,
        )
        df = runner.run(
            policies=["order_up_to"],
            forecasters=["naive", ("moving_average", {"window": 5})],
        )
        forecasters_in_result = df["forecaster"].unique()
        assert "naive" in forecasters_in_result
        assert "moving_average" in forecasters_in_result

    def test_compare_baseline(self):
        runner = BenchmarkRunner(
            chain_config="consumer_2tier",
            demand="semiconductor_ar1",
            T=52,
            N=2,
            seed=42,
        )
        df = runner.run(
            policies=["order_up_to", ("proportional_out", {"alpha": 0.5})],
        )
        comparison = runner.compare(df, baseline="order_up_to")
        assert "pct_change" in comparison.columns
        # Baseline rows should have 0% change
        baseline_rows = comparison[comparison["policy"] == "order_up_to"]
        assert all(baseline_rows["pct_change"].abs() < 1e-10)

    def test_export_csv(self):
        runner = BenchmarkRunner(
            chain_config="consumer_2tier",
            demand="semiconductor_ar1",
            T=52,
            N=2,
            seed=42,
        )
        df = runner.run(policies=["order_up_to"])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            runner.export_csv(df, path)
            loaded = pd.read_csv(path)
            assert len(loaded) == len(df)
        finally:
            os.unlink(path)

    def test_export_latex(self):
        runner = BenchmarkRunner(
            chain_config="consumer_2tier",
            demand="semiconductor_ar1",
            T=52,
            N=2,
            seed=42,
        )
        df = runner.run(policies=["order_up_to"])
        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as f:
            path = f.name
        try:
            runner.export_latex(df, path, caption="Test", label="tab:test")
            with open(path) as f:
                content = f.read()
            assert "\\begin{table}" in content or "tabular" in content
        finally:
            os.unlink(path)


class TestReport:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame([
            {"policy": "out", "forecaster": "naive", "echelon": "E1", "metric": "BWR", "value": 1.5},
            {"policy": "out", "forecaster": "naive", "echelon": "E1", "metric": "TC", "value": 100.0},
            {"policy": "pout", "forecaster": "naive", "echelon": "E1", "metric": "BWR", "value": 1.2},
            {"policy": "pout", "forecaster": "naive", "echelon": "E1", "metric": "TC", "value": 110.0},
        ])

    def test_to_latex(self, sample_df):
        result = to_latex(sample_df)
        assert "tabular" in result

    def test_to_markdown(self, sample_df):
        result = to_markdown(sample_df)
        # Should contain table data regardless of format (markdown or fallback)
        assert "BWR" in result
        assert "out" in result


class TestEndToEnd:
    def test_full_pipeline(self):
        """Full pipeline: init -> run -> compare -> export."""
        runner = BenchmarkRunner(
            chain_config="consumer_2tier",
            demand="beer_game",
            T=52,
            N=1,  # Deterministic demand
            seed=42,
        )
        df = runner.run(
            policies=[
                "order_up_to",
                ("proportional_out", {"alpha": 0.5}),
            ],
            forecasters=["naive"],
            metrics=["BWR", "FILL_RATE", "TC"],
        )

        # Should have: 2 policies x 1 forecaster x 2 echelons x 3 metrics = 12 rows
        assert len(df) == 12
        assert set(df["policy"].unique()) == {"order_up_to", "proportional_out"}
        assert set(df["echelon"].unique()) == {"E1", "E2"}
        assert set(df["metric"].unique()) == {"BWR", "FILL_RATE", "TC"}

        # Compare
        comparison = runner.compare(df, baseline="order_up_to")
        assert "pct_change" in comparison.columns
