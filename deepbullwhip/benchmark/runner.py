"""BenchmarkRunner: standardized bullwhip benchmarking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from deepbullwhip._types import SimulationResult, TimeSeries
from deepbullwhip.benchmark.configs import PREDEFINED_CONFIGS
from deepbullwhip.chain.config import EchelonConfig
from deepbullwhip.chain.echelon import SupplyChainEchelon
from deepbullwhip.chain.serial import SerialSupplyChain
from deepbullwhip.cost.newsvendor import NewsvendorCost
from deepbullwhip.demand.base import DemandGenerator
from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.registry import get, get_class

# Ensure metrics are registered when benchmark module is imported
import deepbullwhip.metrics  # noqa: F401


class BenchmarkRunner:
    """Run standardized bullwhip benchmarks.

    Parameters
    ----------
    chain_config : str or list[EchelonConfig]
        Chain name ("semiconductor_4tier", "beer_game", "consumer_2tier")
        or explicit config list.
    demand : str or DemandGenerator
        Demand name ("semiconductor_ar1", "beer_game", "arma")
        or explicit generator instance.
    T : int
        Number of time periods per simulation.
    N : int
        Number of Monte Carlo paths (1 for deterministic demand).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        chain_config: str | list[EchelonConfig] = "semiconductor_4tier",
        demand: str | DemandGenerator = "semiconductor_ar1",
        T: int = 156,
        N: int = 1000,
        seed: int = 42,
    ) -> None:
        if isinstance(chain_config, str):
            if chain_config not in PREDEFINED_CONFIGS:
                raise KeyError(
                    f"Unknown chain config '{chain_config}'. "
                    f"Available: {list(PREDEFINED_CONFIGS.keys())}"
                )
            self.configs = PREDEFINED_CONFIGS[chain_config]
        else:
            self.configs = chain_config

        if isinstance(demand, str):
            self.demand_gen = get("demand", demand)
        else:
            self.demand_gen = demand

        self.T = T
        self.N = N
        self.seed = seed

    def _build_chain(
        self,
        policy_name: str,
        policy_kwargs: dict[str, Any] | None = None,
    ) -> SerialSupplyChain:
        """Build a SerialSupplyChain with the given policy for all echelons."""
        policy_cls = get_class("policy", policy_name)
        echelons = []
        for cfg in self.configs:
            total_h = cfg.holding_cost + cfg.depreciation_rate
            # Build policy kwargs from config
            kwargs: dict[str, Any] = {"lead_time": cfg.lead_time, "service_level": cfg.service_level}
            if policy_kwargs:
                kwargs.update(policy_kwargs)

            # Handle policies that don't take lead_time/service_level
            try:
                policy = policy_cls(**kwargs)
            except TypeError:
                # Fallback: try with just policy_kwargs (e.g., ConstantOrderPolicy)
                policy = policy_cls(**(policy_kwargs or {}))

            cost_fn = NewsvendorCost(
                holding_cost=total_h, backorder_cost=cfg.backorder_cost
            )
            echelons.append(
                SupplyChainEchelon(
                    name=cfg.name,
                    lead_time=cfg.lead_time,
                    policy=policy,
                    cost_fn=cost_fn,
                    initial_inventory=cfg.initial_inventory,
                )
            )
        return SerialSupplyChain(echelons=echelons)

    def run(
        self,
        policies: list[str | tuple[str, dict[str, Any]]],
        forecasters: list[str | tuple[str, dict[str, Any]]] | None = None,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run all (policy x forecaster) combinations and compute metrics.

        Parameters
        ----------
        policies : list
            Policy names or (name, kwargs) tuples.
            E.g. ["order_up_to", ("proportional_out", {"alpha": 0.3})]
        forecasters : list or None
            Forecaster names or (name, kwargs) tuples. Default: ["naive"].
        metrics : list[str] or None
            Metric names to compute. Default: ["BWR", "CUM_BWR", "FILL_RATE", "TC"].

        Returns
        -------
        pd.DataFrame
            Columns: policy, forecaster, echelon, metric, value
        """
        if forecasters is None:
            forecasters = ["naive"]
        if metrics is None:
            metrics = ["BWR", "CUM_BWR", "FILL_RATE", "TC"]

        rows: list[dict[str, Any]] = []

        for policy_spec in policies:
            if isinstance(policy_spec, str):
                policy_name = policy_spec
                policy_kwargs: dict[str, Any] | None = None
            else:
                policy_name, policy_kwargs = policy_spec

            chain = self._build_chain(policy_name, policy_kwargs)

            for forecaster_spec in forecasters:
                if isinstance(forecaster_spec, str):
                    forecaster_name = forecaster_spec
                    forecaster_kwargs: dict[str, Any] = {}
                else:
                    forecaster_name, forecaster_kwargs = forecaster_spec

                forecaster: Forecaster = get("forecaster", forecaster_name, **forecaster_kwargs)

                # Run N Monte Carlo paths
                metric_accum: dict[tuple[int, str], list[float]] = {}

                rng = np.random.default_rng(self.seed)
                for path_idx in range(self.N):
                    path_seed = int(rng.integers(0, 2**31))
                    demand = self.demand_gen.generate(T=self.T, seed=path_seed)
                    fm, fs = forecaster.generate_forecasts(demand)

                    # Reset chain and simulate
                    chain.reset()
                    result = chain.simulate(demand, fm, fs)

                    # Compute metrics per echelon
                    for k in range(len(result.echelon_results)):
                        for metric_name in metrics:
                            metric_cls = get_class("metric", metric_name)
                            value = metric_cls.compute(result, demand, echelon=k)
                            key = (k, metric_name)
                            if key not in metric_accum:
                                metric_accum[key] = []
                            metric_accum[key].append(value)

                # Average across paths
                for (k, metric_name), values in metric_accum.items():
                    rows.append({
                        "policy": policy_name,
                        "forecaster": forecaster_name,
                        "echelon": f"E{k + 1}",
                        "metric": metric_name,
                        "value": float(np.mean(values)),
                    })

        return pd.DataFrame(rows)

    def compare(
        self,
        results: pd.DataFrame,
        baseline: str = "order_up_to",
    ) -> pd.DataFrame:
        """Compute percentage change vs a baseline policy.

        Parameters
        ----------
        results : pd.DataFrame
            Output of run().
        baseline : str
            Policy name to use as baseline.

        Returns
        -------
        pd.DataFrame
            Same structure with added 'pct_change' column.
        """
        baseline_df = results[results["policy"] == baseline].copy()
        baseline_map = baseline_df.set_index(
            ["forecaster", "echelon", "metric"]
        )["value"].to_dict()

        rows = []
        for _, row in results.iterrows():
            key = (row["forecaster"], row["echelon"], row["metric"])
            base_val = baseline_map.get(key)
            if base_val is not None and base_val != 0:
                pct = (row["value"] - base_val) / abs(base_val) * 100
            else:
                pct = 0.0
            new_row = row.to_dict()
            new_row["pct_change"] = pct
            rows.append(new_row)

        return pd.DataFrame(rows)

    def export_latex(
        self,
        df: pd.DataFrame,
        path: str,
        caption: str = "",
        label: str = "",
    ) -> None:
        """Export results as LaTeX booktabs table to a file.

        Parameters
        ----------
        df : pd.DataFrame
            Results from run().
        path : str
            Output file path.
        caption, label : str
            LaTeX caption and label.
        """
        from deepbullwhip.benchmark.report import to_latex

        latex_str = to_latex(df, caption=caption, label=label)
        with open(path, "w") as f:
            f.write(latex_str)

    def export_csv(self, df: pd.DataFrame, path: str) -> None:
        """Export results to CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Results from run().
        path : str
            Output file path.
        """
        df.to_csv(path, index=False)
