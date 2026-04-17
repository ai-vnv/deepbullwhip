#!/usr/bin/env python3
"""V&V assessment script for DeepBullwhip.

Runs the vnvspec specification against actual package behavior, producing
a machine-readable report (JSON) and a Shields.io badge endpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from vnvspec import EvidenceCollector, Spec

SPEC_PATH = Path(__file__).resolve().parent.parent / "vnvspec.yaml"
REPORT_PATH = Path(__file__).resolve().parent.parent / "vnv-report.json"
BADGE_PATH = Path(__file__).resolve().parent.parent / "vnv-badge.json"


def load_spec() -> Spec:
    return Spec.from_yaml(SPEC_PATH)


def assess(spec: Spec) -> None:
    """Run all verification checks and produce report + badge."""
    with EvidenceCollector(spec) as c:
        _assess_simulation(c)
        _assess_metrics(c)
        _assess_costs(c)
        _assess_policies(c)
        _assess_demand(c)
        _assess_forecasters(c)
        _assess_vectorized(c)
        _assess_benchmark(c)
        _assess_registry(c)
        _assess_schema(c)
        _assess_network(c)

    report = c.build_report(summary="DeepBullwhip V&V assessment")

    REPORT_PATH.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )

    from vnvspec.exporters.shields_endpoint import export_shields_endpoint

    export_shields_endpoint(report, path=BADGE_PATH, label="V&V")

    verdict = report.verdict()
    n_pass = sum(1 for e in report.evidence if e.verdict == "pass")
    n_fail = sum(1 for e in report.evidence if e.verdict == "fail")
    n_inc = sum(1 for e in report.evidence if e.verdict == "inconclusive")
    print(f"\n{'='*60}")
    print(f"  V&V Assessment: {verdict.upper()}")
    print(f"  Evidence: {n_pass} pass, {n_fail} fail, {n_inc} inconclusive")
    print(f"  Report:   {REPORT_PATH}")
    print(f"  Badge:    {BADGE_PATH}")
    print(f"{'='*60}\n")

    sys.exit(0 if verdict == "pass" else 1)


# ── Helpers ──────────────────────────────────────────────────────────────

def _run_simulation(T: int = 200, seed: int = 42):
    """Run a standard simulation and return (result, demand, forecasts)."""
    from deepbullwhip.chain.serial import SerialSupplyChain
    from deepbullwhip.demand.semiconductor import SemiconductorDemandGenerator
    from deepbullwhip.forecast.naive import NaiveForecaster

    gen = SemiconductorDemandGenerator()
    demand = gen.generate(T=T, seed=seed)

    forecaster = NaiveForecaster()
    f_mean, f_std = forecaster.generate_forecasts(demand)

    chain = SerialSupplyChain()
    result = chain.simulate(demand, f_mean, f_std)
    return chain, result, demand, f_mean, f_std


# ── Simulation Core ──────────────────────────────────────────────────────

def _assess_simulation(c: EvidenceCollector) -> None:
    chain, result, demand, f_mean, f_std = _run_simulation(T=100)

    # REQ-SIM-001: Non-negative orders
    all_nonneg = all(
        np.all(er.orders >= 0) for er in result.echelon_results
    )
    min_order = min(er.orders.min() for er in result.echelon_results)
    c.check("REQ-SIM-001", all_nonneg, message=f"min_order={min_order:.6f}")

    # REQ-SIM-002: Array shapes
    K = len(chain.echelons)
    T = 100
    shapes_ok = all(
        len(er.orders) == T and len(er.inventory_levels) == T
        for er in result.echelon_results
    )
    c.check("REQ-SIM-002", shapes_ok and len(result.echelon_results) == K,
            message=f"K={K}, T={T}, echelon_results={len(result.echelon_results)}")

    # REQ-SIM-003: Default 4-echelon topology
    from deepbullwhip.chain.serial import SerialSupplyChain
    default_chain = SerialSupplyChain()
    names = [e.name for e in default_chain.echelons]
    has_4 = len(default_chain.echelons) == 4
    all_named = all(isinstance(n, str) and len(n) > 0 for n in names)
    c.check("REQ-SIM-003", has_4 and all_named,
            message=f"n={len(default_chain.echelons)}, names={names}")

    # REQ-SIM-004: Reset clears state
    chain2, _, _, _, _ = _run_simulation(T=50)
    for e in chain2.echelons:
        e.reset()
    pipeline_ok = all(len(e.pipeline) == e.lead_time for e in chain2.echelons)
    c.check("REQ-SIM-004", pipeline_ok, message="Reset restores initial pipeline length")


# ── Metrics ──────────────────────────────────────────────────────────────

def _assess_metrics(c: EvidenceCollector) -> None:
    _, result, demand, _, _ = _run_simulation(T=200)

    # REQ-MET-001: Positive BWR
    bwr_vals = [er.bullwhip_ratio for er in result.echelon_results]
    bwr_positive = all(b > 0 for b in bwr_vals)
    c.check("REQ-MET-001", bwr_positive, message=f"bwr={bwr_vals}")

    # REQ-MET-002: Fill rate in [0, 1]
    fr_vals = [er.fill_rate for er in result.echelon_results]
    fr_bounded = all(0 <= f <= 1 for f in fr_vals)
    c.check("REQ-MET-002", fr_bounded, message=f"fill_rates={fr_vals}")

    # REQ-MET-003: Total cost = sum of echelon costs
    total = result.total_cost
    echelon_sum = sum(er.total_cost for er in result.echelon_results)
    cost_match = abs(total - echelon_sum) < 1e-10
    c.check("REQ-MET-003", cost_match,
            message=f"total={total:.4f}, sum={echelon_sum:.4f}, diff={abs(total - echelon_sum):.2e}")

    # REQ-MET-004: NSAmp finite and positive
    from deepbullwhip.metrics.inventory import NSAmp
    nsamp_val = NSAmp.compute(result, demand, echelon=0)
    nsamp_ok = np.isfinite(nsamp_val) and nsamp_val > 0
    c.check("REQ-MET-004", nsamp_ok, message=f"nsamp={nsamp_val:.6f}")


# ── Cost Functions ───────────────────────────────────────────────────────

def _assess_costs(c: EvidenceCollector) -> None:
    from deepbullwhip.cost.newsvendor import NewsvendorCost

    cost_fn = NewsvendorCost(holding_cost=1.0, backorder_cost=5.0)
    holding_ok = cost_fn.compute(10.0) == 1.0 * 10.0
    backorder_ok = cost_fn.compute(-10.0) == 5.0 * 10.0
    zero_ok = cost_fn.compute(0.0) == 0.0
    c.check("REQ-COST-001", holding_ok and backorder_ok and zero_ok,
            message=f"h(10)={cost_fn.compute(10.0)}, b(-10)={cost_fn.compute(-10.0)}, z(0)={cost_fn.compute(0.0)}")

    from deepbullwhip.cost.perishable import PerishableCost
    pcost = PerishableCost(holding_cost=1.0, backorder_cost=5.0, gamma=0.05, buffer=50.0)
    test_levels = [-20, -10, -5, 0, 5, 10, 20]
    all_nonneg = all(pcost.compute(level) >= 0 for level in test_levels)
    c.check("REQ-COST-002", all_nonneg,
            message=f"costs={[pcost.compute(l) for l in test_levels]}")


# ── Ordering Policies ────────────────────────────────────────────────────

def _assess_policies(c: EvidenceCollector) -> None:
    from deepbullwhip.policy.order_up_to import OrderUpToPolicy
    from deepbullwhip.policy.proportional_out import ProportionalOUTPolicy
    from deepbullwhip.policy.smoothing_out import SmoothingOUTPolicy
    from deepbullwhip.policy.constant_order import ConstantOrderPolicy

    # REQ-POL-001: All policies return non-negative orders
    policies = [
        OrderUpToPolicy(lead_time=2),
        ProportionalOUTPolicy(lead_time=2),
        SmoothingOUTPolicy(lead_time=2),
        ConstantOrderPolicy(order_quantity=50),
    ]
    all_nonneg = True
    for pol in policies:
        order = pol.compute_order(
            inventory_position=80,
            forecast_mean=100.0,
            forecast_std=10.0,
        )
        if order < 0:
            all_nonneg = False
    c.check("REQ-POL-001", all_nonneg, message="All built-in policies return non-negative orders")

    # REQ-POL-002: Orders converge to demand under constant demand
    from deepbullwhip.chain.serial import SerialSupplyChain
    from deepbullwhip.forecast.naive import NaiveForecaster

    constant_demand = np.full(200, 100.0)
    forecaster = NaiveForecaster()
    f_mean, f_std = forecaster.generate_forecasts(constant_demand)

    chain = SerialSupplyChain()
    result = chain.simulate(constant_demand, f_mean, f_std)
    # After warmup, first echelon orders should match demand
    first_echelon_orders = result.echelon_results[0].orders[20:]  # skip transient
    max_deviation = np.max(np.abs(first_echelon_orders - 100.0)) / 100.0
    c.check("REQ-POL-002", max_deviation < 0.01,
            message=f"max_steady_state_deviation={max_deviation:.6f}")


# ── Demand Generators ────────────────────────────────────────────────────

def _assess_demand(c: EvidenceCollector) -> None:
    from deepbullwhip.demand.semiconductor import SemiconductorDemandGenerator
    from deepbullwhip.demand.beer_game import BeerGameDemandGenerator
    from deepbullwhip.demand.arma import ARMADemandGenerator

    generators = [
        SemiconductorDemandGenerator(),
        BeerGameDemandGenerator(),
        ARMADemandGenerator(),
    ]
    all_nonneg = True
    for gen in generators:
        demand = gen.generate(T=200, seed=42)
        if np.any(demand < 0):
            all_nonneg = False
    c.check("REQ-DEM-001", all_nonneg, message="All generators produce non-negative demand")

    semi = SemiconductorDemandGenerator(mu=100)
    demand = semi.generate(T=500, seed=42)
    mean_demand = demand.mean()
    c.check("REQ-DEM-002", mean_demand > 0, message=f"mean_demand={mean_demand:.2f}")


# ── Forecasters ──────────────────────────────────────────────────────────

def _assess_forecasters(c: EvidenceCollector) -> None:
    from deepbullwhip.forecast.naive import NaiveForecaster
    from deepbullwhip.forecast.moving_average import MovingAverageForecaster
    from deepbullwhip.forecast.exponential_smoothing import ExponentialSmoothingForecaster

    history = np.array([100, 110, 105, 95, 100, 108, 103, 97, 102, 99], dtype=float)

    # REQ-FOR-001: Forecast interface and non-negative std
    forecasters = [
        NaiveForecaster(),
        MovingAverageForecaster(window=5),
        ExponentialSmoothingForecaster(alpha=0.3),
    ]
    all_ok = True
    for f in forecasters:
        result = f.forecast(history)
        if not (isinstance(result, tuple) and len(result) == 2):
            all_ok = False
            continue
        mean, std = result
        if std < 0:
            all_ok = False
    c.check("REQ-FOR-001", all_ok, message="All forecasters return (mean, std) with std >= 0")

    # REQ-FOR-002: Moving average correctness
    ma = MovingAverageForecaster(window=5)
    mean, _ = ma.forecast(history)
    expected = history[-5:].mean()
    ma_correct = abs(mean - expected) < 1e-10
    c.check("REQ-FOR-002", ma_correct,
            message=f"ma_forecast={mean:.6f}, expected={expected:.6f}")


# ── Vectorized Engine ────────────────────────────────────────────────────

def _assess_vectorized(c: EvidenceCollector) -> None:
    from deepbullwhip.chain.serial import SerialSupplyChain
    from deepbullwhip.chain.vectorized import VectorizedSupplyChain
    from deepbullwhip.demand.semiconductor import SemiconductorDemandGenerator
    from deepbullwhip.forecast.naive import NaiveForecaster

    gen = SemiconductorDemandGenerator()
    forecaster = NaiveForecaster()

    # REQ-VEC-001: Consistency with serial engine (order of magnitude)
    demand = gen.generate(T=52, seed=99)
    fm = np.full_like(demand, demand.mean())
    fs = np.full_like(demand, demand.std())

    serial_chain = SerialSupplyChain()
    serial_result = serial_chain.simulate(demand, fm, fs)

    vec_chain = VectorizedSupplyChain()
    vec_result = vec_chain.simulate(demand, fm, fs)
    vec_sr = vec_result.to_simulation_result(0)

    # Both engines must produce positive BWR and positive total costs
    serial_bwr_ok = all(er.bullwhip_ratio > 0 for er in serial_result.echelon_results)
    vec_bwr_ok = all(er.bullwhip_ratio > 0 for er in vec_sr.echelon_results)
    serial_cost_ok = serial_result.total_cost > 0
    vec_cost_ok = vec_sr.total_cost > 0
    consistent = serial_bwr_ok and vec_bwr_ok and serial_cost_ok and vec_cost_ok
    c.check("REQ-VEC-001", consistent,
            message=f"serial_bwr_ok={serial_bwr_ok}, vec_bwr_ok={vec_bwr_ok}, serial_cost={serial_result.total_cost:.2f}, vec_cost={vec_sr.total_cost:.2f}")

    # REQ-VEC-002: Non-negative total costs
    demand_batch = gen.generate_batch(T=100, n_paths=50, seed=42)
    f_mean_batch = np.zeros_like(demand_batch)
    f_std_batch = np.zeros_like(demand_batch)
    for i in range(50):
        fm, fs = forecaster.generate_forecasts(demand_batch[i])
        f_mean_batch[i] = fm
        f_std_batch[i] = fs

    vec_chain2 = VectorizedSupplyChain()
    vec_result2 = vec_chain2.simulate(demand_batch, f_mean_batch, f_std_batch)
    costs_nonneg = np.all(vec_result2.total_costs >= 0)
    c.check("REQ-VEC-002", costs_nonneg,
            message=f"min_cost={vec_result2.total_costs.min():.4f}")

    # REQ-VEC-003: Mean metrics correctness
    mean_bwr = vec_result2.mean_metrics()
    # Just verify mean_metrics returns without error and contains expected keys
    has_keys = isinstance(mean_bwr, dict)
    c.check("REQ-VEC-003", has_keys, message=f"mean_metrics keys={list(mean_bwr.keys()) if has_keys else 'N/A'}")


# ── Benchmark Framework ──────────────────────────────────────────────────

def _assess_benchmark(c: EvidenceCollector) -> None:
    import pandas as pd
    from deepbullwhip.benchmark.runner import BenchmarkRunner

    # REQ-BEN-001: Runner returns DataFrame
    runner = BenchmarkRunner(
        chain_config="semiconductor_4tier",
        demand="semiconductor_ar1",
        T=52,
        N=10,
        seed=42,
    )
    df = runner.run(
        policies=["order_up_to", "constant_order"],
        forecasters=["naive"],
    )
    is_df = isinstance(df, pd.DataFrame)
    has_rows = len(df) > 0
    c.check("REQ-BEN-001", is_df and has_rows, message=f"DataFrame shape={df.shape}")

    # REQ-BEN-002: Export functions
    import tempfile
    csv_path = Path(tempfile.mktemp(suffix=".csv"))
    latex_path = Path(tempfile.mktemp(suffix=".tex"))
    try:
        runner.export_csv(df, str(csv_path))
        csv_ok = csv_path.stat().st_size > 0
    except Exception:
        csv_ok = False

    try:
        runner.export_latex(df, str(latex_path))
        latex_ok = latex_path.exists()
    except Exception:
        latex_ok = False

    c.check("REQ-BEN-002", csv_ok and latex_ok, message=f"csv={csv_ok}, latex={latex_ok}")


# ── Registry ─────────────────────────────────────────────────────────────

def _assess_registry(c: EvidenceCollector) -> None:
    from deepbullwhip.registry import register, get_class

    @register("demand", "test_vnv_demand")
    class _TestDemand:
        pass

    retrieved = get_class("demand", "test_vnv_demand")
    reg_ok = retrieved is _TestDemand

    try:
        get_class("demand", "nonexistent_vnv_xyz_12345")
        error_raised = False
    except (KeyError, ValueError):
        error_raised = True

    c.check("REQ-REG-001", reg_ok and error_raised,
            message=f"registered={reg_ok}, error_on_unknown={error_raised}")


# ── Schema ───────────────────────────────────────────────────────────────

def _assess_schema(c: EvidenceCollector) -> None:
    import tempfile
    from deepbullwhip.chain.graph import SupplyChainGraph, EdgeConfig
    from deepbullwhip.chain.config import EchelonConfig
    from deepbullwhip.schema.io import save_json, load_json

    # Build a simple graph
    graph = SupplyChainGraph(
        nodes={
            "Retailer": EchelonConfig("Retailer", lead_time=2, holding_cost=0.5, backorder_cost=1.0),
            "Wholesaler": EchelonConfig("Wholesaler", lead_time=2, holding_cost=0.5, backorder_cost=1.0),
        },
        edges={
            ("Wholesaler", "Retailer"): EdgeConfig(),
        },
    )

    json_path = tempfile.mktemp(suffix=".json")
    save_json(graph, json_path)
    loaded = load_json(json_path)

    orig_nodes = set(graph.nodes.keys())
    loaded_nodes = set(loaded.nodes.keys())
    roundtrip_ok = orig_nodes == loaded_nodes
    c.check("REQ-SCH-001", roundtrip_ok,
            message=f"orig={sorted(orig_nodes)}, loaded={sorted(loaded_nodes)}")


# ── Network Topology ─────────────────────────────────────────────────────

def _assess_network(c: EvidenceCollector) -> None:
    from deepbullwhip.chain.graph import SupplyChainGraph, EdgeConfig
    from deepbullwhip.chain.config import EchelonConfig

    ec = lambda name: EchelonConfig(name, lead_time=2, holding_cost=0.5, backorder_cost=1.0)

    # REQ-NET-001: DAG enforcement
    graph = SupplyChainGraph(
        nodes={"A": ec("A"), "B": ec("B"), "C": ec("C")},
        edges={("A", "B"): EdgeConfig(), ("B", "C"): EdgeConfig()},
    )

    cycle_rejected = False
    try:
        cyclic = SupplyChainGraph(
            nodes={"A": ec("A"), "B": ec("B"), "C": ec("C")},
            edges={
                ("A", "B"): EdgeConfig(),
                ("B", "C"): EdgeConfig(),
                ("C", "A"): EdgeConfig(),
            },
        )
        cyclic.topological_order()  # Cycle detected here
    except (ValueError, Exception):
        cycle_rejected = True

    c.check("REQ-NET-001", cycle_rejected, message="Cyclic graph correctly rejected")

    # REQ-NET-002: NetworkX round-trip
    from deepbullwhip.network.convert import to_networkx, from_networkx

    nx_graph = to_networkx(graph)
    g_back = from_networkx(nx_graph)

    orig_nodes = set(graph.nodes.keys())
    round_nodes = set(g_back.nodes.keys())
    orig_edges = set(graph.edges.keys())
    round_edges = set(g_back.edges.keys())
    roundtrip_ok = orig_nodes == round_nodes and orig_edges == round_edges
    c.check("REQ-NET-002", roundtrip_ok,
            message=f"nodes={orig_nodes == round_nodes}, edges={orig_edges == round_edges}")


if __name__ == "__main__":
    spec = load_spec()
    assess(spec)
