"""
Generate the official deepbullwhip benchmark leaderboard.

Usage:
    python benchmarks/run_leaderboard.py [--output docs/LEADERBOARD.md] [--sort-by TC]

This script discovers all registered components and runs them through
a standardized benchmark protocol. Results are deterministic (seed=966).
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time

import pandas as pd

# Import all modules so @register decorators fire
import deepbullwhip.demand  # noqa: F401
import deepbullwhip.policy  # noqa: F401
import deepbullwhip.forecast  # noqa: F401
import deepbullwhip.metrics  # noqa: F401
from deepbullwhip.registry import list_registered
from deepbullwhip.benchmark import BenchmarkRunner

# ── Fixed benchmark protocol ─────────────────────────────────────────
SEED = 966
T = 156
N = 1000
CHAIN = "semiconductor_4tier"
DEMAND = "semiconductor_ar1"

# ── Component metadata: contributor + reference ──────────────────────
# Add an entry here when contributing a new component.
# Format: "registry_name": {"contributor": "Name", "reference": "DOI or paper"}
COMPONENT_META: dict[str, dict[str, str]] = {
    # Forecasters
    "naive": {
        "contributor": "M. Arief",
        "reference": "",
    },
    "moving_average": {
        "contributor": "M. Arief",
        "reference": "",
    },
    "exponential_smoothing": {
        "contributor": "M. Arief",
        "reference": "",
    },
    "deepar": {
        "contributor": "M. Arief",
        "reference": '<a href="https://doi.org/10.1016/j.ijforecast.2019.07.001">Salinas et al. 2020</a>',
    },
    # Policies
    "order_up_to": {
        "contributor": "M. Arief",
        "reference": '<a href="https://doi.org/10.1287/mnsc.46.3.436.12069">Chen et al. 2000</a>',
    },
    "proportional_out": {
        "contributor": "M. Arief",
        "reference": '<a href="https://doi.org/10.1080/0020754031000114743">Disney &amp; Towill 2003</a>',
    },
    "smoothing_out": {
        "contributor": "M. Arief",
        "reference": "",
    },
    "constant_order": {
        "contributor": "M. Arief",
        "reference": "",
    },
    # Demand generators
    "semiconductor_ar1": {
        "contributor": "M. Arief",
        "reference": '<a href="https://www.wsts.org">WSTS</a>',
    },
    "beer_game": {
        "contributor": "M. Arief",
        "reference": '<a href="https://doi.org/10.1287/mnsc.35.3.321">Sterman 1989</a>',
    },
    "arma": {
        "contributor": "M. Arief",
        "reference": "",
    },
    "replay": {
        "contributor": "M. Arief",
        "reference": "",
    },
}


def _get_meta(name: str) -> tuple[str, str]:
    """Return (contributor, reference) for a component name."""
    meta = COMPONENT_META.get(name, {})
    return meta.get("contributor", ""), meta.get("reference", "")


def run_forecaster_sweep() -> pd.DataFrame:
    """Sweep A: all forecasters under OUT policy."""
    runner = BenchmarkRunner(CHAIN, DEMAND, T=T, N=N, seed=SEED)
    forecasters = list_registered("forecaster")
    results = runner.run(
        policies=["order_up_to"],
        forecasters=list(forecasters),
        metrics=["CUM_BWR", "FILL_RATE", "TC"],
    )
    max_ech = results["echelon"].max()
    e_top = results[results["echelon"] == max_ech]
    return e_top.pivot_table(index="forecaster", columns="metric", values="value")


def run_policy_sweep() -> pd.DataFrame:
    """Sweep B: all policies under naive forecaster."""
    runner = BenchmarkRunner(CHAIN, DEMAND, T=T, N=N, seed=SEED)
    policies = list_registered("policy")
    results = runner.run(
        policies=list(policies),
        forecasters=["naive"],
        metrics=["CUM_BWR", "FILL_RATE", "TC", "NSAmp"],
    )
    max_ech = results["echelon"].max()
    e_top = results[results["echelon"] == max_ech]
    return e_top.pivot_table(index="policy", columns="metric", values="value")


def run_demand_sweep() -> pd.DataFrame:
    """Sweep C: all demand generators under OUT + naive."""
    demands = list_registered("demand")
    rows = []
    for dname in demands:
        if dname == "replay":
            continue  # skip replay -- requires external data
        try:
            runner = BenchmarkRunner(CHAIN, dname, T=T, N=N, seed=SEED)
            results = runner.run(
                policies=["order_up_to"],
                forecasters=["naive"],
                metrics=["CUM_BWR", "TC"],
            )
            max_ech = results["echelon"].max()
            e_top = results[results["echelon"] == max_ech]
            for _, row in e_top.iterrows():
                rows.append({
                    "demand": dname,
                    "metric": row["metric"],
                    "value": row["value"],
                })
        except Exception as e:
            rows.append({"demand": dname, "metric": "error", "value": str(e)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.pivot_table(index="demand", columns="metric", values="value")


def format_markdown_table(
    df: pd.DataFrame,
    title: str,
    sort_by: str = "TC",
    ascending: bool = True,
) -> str:
    """Convert DataFrame to a sortable HTML table with title."""
    if df.empty:
        return f"### {title}\n\n_No results._\n"

    sort_col = sort_by if sort_by in df.columns else ("TC" if "TC" in df.columns else df.columns[0])
    df_sorted = df.sort_values(sort_col, ascending=ascending)
    cols = df.columns.tolist()

    lines = [f"### {title}\n"]

    # Build HTML table with sortable headers (JS at bottom of file)
    lines.append('<table class="sortable">')

    # Header row
    header_cells = ['<th>Component</th>', '<th>Contributor</th>', '<th>Reference</th>']
    for c in cols:
        header_cells.append(f'<th>{c}</th>')
    lines.append('  <thead><tr>' + ''.join(header_cells) + '</tr></thead>')

    # Body rows
    lines.append('  <tbody>')
    for idx, row in df_sorted.iterrows():
        contributor, reference = _get_meta(str(idx))
        cells = [
            f'<td><code>{idx}</code></td>',
            f'<td>{contributor}</td>',
            f'<td>{reference}</td>',
        ]
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f'<td>{v:,.1f}</td>')
            else:
                cells.append(f'<td>{v}</td>')
        lines.append('    <tr>' + ''.join(cells) + '</tr>')
    lines.append('  </tbody>')
    lines.append('</table>\n')

    return "\n".join(lines)


SORTABLE_JS = """
<script>
// Minimal sortable tables: click any <th> to sort
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("table.sortable th").forEach(function (th) {
    th.style.cursor = "pointer";
    th.addEventListener("click", function () {
      var table = th.closest("table");
      var tbody = table.querySelector("tbody");
      var rows = Array.from(tbody.querySelectorAll("tr"));
      var idx = Array.from(th.parentNode.children).indexOf(th);
      var asc = th.dataset.sortDir !== "asc";
      th.dataset.sortDir = asc ? "asc" : "desc";
      rows.sort(function (a, b) {
        var av = a.children[idx].textContent.replace(/,/g, "");
        var bv = b.children[idx].textContent.replace(/,/g, "");
        var an = parseFloat(av), bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      });
      rows.forEach(function (r) { tbody.appendChild(r); });
    });
  });
});
</script>
"""


def main(output_path: str = "docs/LEADERBOARD.md", sort_by: str = "TC") -> None:
    t_start = time.perf_counter()
    sections = []
    sections.append("# deepbullwhip Benchmark Leaderboard\n")
    sections.append("> Auto-generated by `benchmarks/run_leaderboard.py`.")
    sections.append(
        f"> Benchmark protocol: chain=`{CHAIN}`, demand=`{DEMAND}`, "
        f"T={T}, N={N:,}, seed={SEED}"
    )
    sections.append(
        "> All results reported at the most upstream echelon (E4)."
    )
    sections.append(
        "> Click any column header to sort.\n"
    )

    print("Running forecaster sweep...")
    fc_df = run_forecaster_sweep()
    sections.append(
        format_markdown_table(
            fc_df, "Forecaster Leaderboard (policy=order_up_to)", sort_by=sort_by
        )
    )

    print("Running policy sweep...")
    pol_df = run_policy_sweep()
    sections.append(
        format_markdown_table(
            pol_df, "Policy Leaderboard (forecaster=naive)", sort_by=sort_by
        )
    )

    print("Running demand generator sweep...")
    dem_df = run_demand_sweep()
    sections.append(
        format_markdown_table(
            dem_df,
            "Demand Generator Leaderboard (policy=order_up_to, forecaster=naive)",
            sort_by=sort_by,
        )
    )

    # Append sortable JS for rendered HTML (MkDocs / GitHub Pages)
    sections.append(SORTABLE_JS)

    md = "\n".join(sections)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)

    elapsed = time.perf_counter() - t_start
    print(f"\nLeaderboard written to {output_path} ({elapsed:.1f}s)")
    print(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the deepbullwhip benchmark leaderboard"
    )
    parser.add_argument("--output", default="docs/LEADERBOARD.md")
    parser.add_argument(
        "--sort-by", default="TC",
        help="Column to sort tables by (default: TC)",
    )
    args = parser.parse_args()
    main(args.output, sort_by=args.sort_by)
