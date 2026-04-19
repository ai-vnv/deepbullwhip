"""
Policy × forecaster leaderboard (composite rank across extended metrics).

Runs registered policies and forecasters through
:class:`~deepbullwhip.benchmark.runner.BenchmarkRunner`, including
components from ``deepbullwhip.ext``. Writes optional CSV/Markdown when
``--out`` is set.

Usage::

    python scripts/run_learning_leaderboard.py
    python scripts/run_learning_leaderboard.py --T 156 --N 20 --out leaderboard
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

import deepbullwhip.ext  # noqa: F401

from deepbullwhip.benchmark.runner import BenchmarkRunner
from deepbullwhip.registry import list_registered

LOWER_BETTER = {
    "BWR",
    "CUM_BWR",
    "NSAmp",
    "TC",
    "RFU",
    "OSR",
    "PeakBWR",
    "ExpectedShortfall",
}

HIGHER_BETTER = {
    "FILL_RATE",
    "InventoryTurnover",
}


def _default_policies() -> list[str]:
    reg = list_registered("policy")
    preferred = [
        "order_up_to",
        "proportional_out",
        "smoothing_out",
        "dqn_beer_game",
        "recurrent_ppo",
        "dcl",
        "e2e_newsvendor",
    ]
    return [p for p in preferred if p in reg]


def _default_forecasters() -> list[str]:
    reg = list_registered("forecaster")
    preferred = [
        "naive",
        "moving_average",
        "exponential_smoothing",
        "deepar",
        "nbeats",
        "tft",
        "lightgbm_quantile",
        "lstm_multistep",
    ]
    return [f for f in preferred if f in reg]


def _default_metrics() -> list[str]:
    reg = list_registered("metric")
    preferred = [
        "BWR",
        "CUM_BWR",
        "NSAmp",
        "FILL_RATE",
        "TC",
        "RFU",
        "OSR",
        "PeakBWR",
        "ExpectedShortfall",
        "InventoryTurnover",
        "DampingRatio",
    ]
    return [m for m in preferred if m in reg]


def build_leaderboard(df: pd.DataFrame, focus_echelon: str = "E4") -> pd.DataFrame:
    last = df[df["echelon"] == focus_echelon].copy()
    return last.pivot_table(
        index=["policy", "forecaster"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()


def rank_by_composite(pivot: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in pivot.columns if c in LOWER_BETTER or c in HIGHER_BETTER]
    if not cols:
        pivot = pivot.copy()
        pivot["composite_rank"] = np.nan
        return pivot

    ranks = []
    for c in cols:
        series = pivot[c].astype(float)
        if c in LOWER_BETTER:
            ranks.append(series.rank(method="average", ascending=True))
        else:
            ranks.append(series.rank(method="average", ascending=False))
    ranks_df = pd.concat(ranks, axis=1)
    pivot = pivot.copy()
    pivot["composite_rank"] = ranks_df.mean(axis=1)
    return pivot.sort_values("composite_rank", ascending=True).reset_index(drop=True)


def to_markdown(pivot: pd.DataFrame, focus_echelon: str, n_paths: int, t_horizon: int) -> str:
    lines = [
        "# Learning leaderboard (deepbullwhip.ext)",
        "",
        f"*4-tier semiconductor chain, focus echelon **{focus_echelon}**, "
        f"N = {n_paths} Monte Carlo paths, T = {t_horizon} periods.*",
        "",
        "Scoring: lower-is-better for BWR / CUM_BWR / NSAmp / TC / RFU / OSR / "
        "PeakBWR / ExpectedShortfall; higher-is-better for FILL_RATE / "
        "InventoryTurnover. `composite_rank` is the mean rank across those "
        "metrics. `DampingRatio` is reported but not scored.",
        "",
    ]

    display = pivot.copy()
    for c in display.columns:
        if display[c].dtype.kind in "fc":
            display[c] = display[c].map(
                lambda v: "–" if pd.isna(v) else (f"{v:,.2f}" if abs(v) >= 1 else f"{v:.3f}")
            )
    lines.append(display.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="deepbullwhip.ext policy×forecaster leaderboard")
    parser.add_argument("--T", type=int, default=156, help="simulation horizon (default: 156)")
    parser.add_argument("--N", type=int, default=20, help="Monte Carlo paths (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focus-echelon", default="E4")
    parser.add_argument(
        "--out",
        default=None,
        help="output prefix; writes <out>.csv, <out>.md, <out>_raw.csv",
    )
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--forecasters", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="*", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    policies = args.policies or _default_policies()
    forecasters = args.forecasters or _default_forecasters()
    metrics = args.metrics or _default_metrics()

    print(f"[leaderboard] policies    = {policies}")
    print(f"[leaderboard] forecasters = {forecasters}")
    print(f"[leaderboard] metrics     = {metrics}")
    print(f"[leaderboard] T={args.T}  N={args.N}  seed={args.seed}")

    runner = BenchmarkRunner(T=args.T, N=args.N, seed=args.seed)
    df = runner.run(
        policies=policies,
        forecasters=forecasters,
        metrics=metrics,
    )

    pivot = build_leaderboard(df, focus_echelon=args.focus_echelon)
    pivot = rank_by_composite(pivot)

    md = to_markdown(pivot, args.focus_echelon, args.N, args.T)
    print("\n" + md)

    if args.out:
        out_dir = os.path.dirname(args.out) or "."
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.out + "_raw.csv", index=False)
        pivot.to_csv(args.out + ".csv", index=False)
        with open(args.out + ".md", "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"[leaderboard] wrote {args.out}.csv / {args.out}.md / {args.out}_raw.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
