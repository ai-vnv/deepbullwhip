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

# ── Human-readable column labels ─────────────────────────────────────
METRIC_LABELS: dict[str, str] = {
    "CUM_BWR": "Cum. BWR",
    "FILL_RATE": "Fill Rate",
    "TC": "Total Cost",
    "NSAmp": "NS Amp.",
}

# ── Component metadata: contributor + reference ──────────────────────
# Add an entry here when contributing a new component.
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

RANK_BADGES = ["&#129351;", "&#129352;", "&#129353;"]  # gold, silver, bronze


def _get_meta(name: str) -> tuple[str, str]:
    """Return (contributor, reference) for a component name."""
    meta = COMPONENT_META.get(name, {})
    return meta.get("contributor", ""), meta.get("reference", "")


def _col_label(col: str) -> str:
    """Return human-readable label for a metric column."""
    return METRIC_LABELS.get(col, col)


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


def format_table(
    df: pd.DataFrame,
    title: str,
    description: str,
    sort_by: str = "TC",
    ascending: bool = True,
) -> str:
    """Convert DataFrame to a styled, sortable HTML table."""
    if df.empty:
        return f"### {title}\n\n_No results._\n"

    sort_col = sort_by if sort_by in df.columns else (
        "TC" if "TC" in df.columns else df.columns[0]
    )
    df_sorted = df.sort_values(sort_col, ascending=ascending)
    cols = df.columns.tolist()

    lines = []
    lines.append(f'<h3>{title}</h3>')
    lines.append(f'<p class="lb-desc">{description}</p>')
    lines.append("")
    lines.append('<table class="lb-table sortable">')

    # Header
    header = ['<th class="lb-rank">#</th>', '<th>Component</th>',
              '<th>Contributor</th>', '<th>Reference</th>']
    for c in cols:
        header.append(f'<th class="lb-metric">{_col_label(c)} <span class="sort-icon"></span></th>')
    lines.append('  <thead><tr>' + ''.join(header) + '</tr></thead>')

    # Body
    lines.append('  <tbody>')
    for rank, (idx, row) in enumerate(df_sorted.iterrows()):
        contributor, reference = _get_meta(str(idx))
        badge = RANK_BADGES[rank] if rank < 3 else str(rank + 1)
        ref_cell = f'<span class="lb-ref">{reference}</span>' if reference else '<span class="lb-ref lb-none">&mdash;</span>'
        cells = [
            f'<td class="lb-rank">{badge}</td>',
            f'<td class="lb-name"><code>{idx}</code></td>',
            f'<td>{contributor}</td>',
            f'<td>{ref_cell}</td>',
        ]
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if c == "FILL_RATE":
                    cells.append(f'<td class="lb-metric">{v:.1%}</td>')
                else:
                    cells.append(f'<td class="lb-metric">{v:,.1f}</td>')
            else:
                cells.append(f'<td class="lb-metric">{v}</td>')
        tr_class = ' class="lb-top3"' if rank < 3 else ""
        lines.append(f'    <tr{tr_class}>' + ''.join(cells) + '</tr>')
    lines.append('  </tbody>')
    lines.append('</table>\n')

    return "\n".join(lines)


# ── Embedded styles + sortable JS ────────────────────────────────────

LEADERBOARD_STYLES = """\
<style>
.lb-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9em;
  margin: 1em 0 2em 0;
}
.lb-table thead {
  background: var(--md-primary-fg-color, #006747);
  color: #fff;
}
.lb-table th {
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
.lb-table th:hover {
  background: var(--md-primary-fg-color--dark, #004d35);
}
.lb-table th .sort-icon::after { content: " \\2195"; font-size: 0.7em; opacity: 0.5; }
.lb-table td {
  padding: 8px 14px;
  border-bottom: 1px solid #e0e0e0;
}
.lb-table tbody tr:hover {
  background: var(--md-primary-fg-color--light, #e8f5f0);
}
.lb-table tr.lb-top3 {
  font-weight: 500;
}
.lb-rank { text-align: center; width: 40px; }
.lb-name code {
  background: #f4f4f4;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.88em;
}
.lb-metric { text-align: right; font-variant-numeric: tabular-nums; }
.lb-ref { font-size: 0.85em; }
.lb-ref.lb-none { opacity: 0.3; }
.lb-desc {
  color: #666;
  font-size: 0.9em;
  margin: -0.5em 0 0.5em 0;
}
.lb-protocol {
  background: #f8f9fa;
  border-left: 3px solid var(--md-primary-fg-color, #006747);
  padding: 12px 16px;
  margin: 1em 0 2em 0;
  font-size: 0.88em;
  color: #555;
  line-height: 1.6;
}
.lb-protocol code {
  background: #eef;
  padding: 1px 5px;
  border-radius: 3px;
}
</style>
"""

SORTABLE_JS = """\
<script>
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("table.sortable th").forEach(function (th) {
    th.addEventListener("click", function () {
      var table = th.closest("table");
      var tbody = table.querySelector("tbody");
      var rows = Array.from(tbody.querySelectorAll("tr"));
      var idx = Array.from(th.parentNode.children).indexOf(th);
      var asc = th.dataset.sortDir !== "asc";
      th.dataset.sortDir = asc ? "asc" : "desc";
      // Reset other headers
      th.parentNode.querySelectorAll("th").forEach(function(h) {
        if (h !== th) delete h.dataset.sortDir;
      });
      rows.sort(function (a, b) {
        var av = a.children[idx].textContent.replace(/,/g, "").replace(/%/g, "");
        var bv = b.children[idx].textContent.replace(/,/g, "").replace(/%/g, "");
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

    # Title
    sections.append("# Benchmark Leaderboard\n")

    # Protocol box
    sections.append(LEADERBOARD_STYLES)
    sections.append(f"""\
<div class="lb-protocol">
<strong>Benchmark protocol</strong><br>
Chain: <code>{CHAIN}</code> &nbsp;|&nbsp;
Demand: <code>{DEMAND}</code> &nbsp;|&nbsp;
T={T} periods &nbsp;|&nbsp;
N={N:,} Monte Carlo paths &nbsp;|&nbsp;
Seed={SEED}<br>
All results reported at the most upstream echelon (E4).
Sorted by Total Cost (lower is better). Click any column header to re-sort.
</div>
""")

    print("Running forecaster sweep...")
    fc_df = run_forecaster_sweep()
    sections.append(
        format_table(
            fc_df,
            "Forecaster Leaderboard",
            "Fixed policy: <code>order_up_to</code> &nbsp;|&nbsp; Fixed demand: <code>semiconductor_ar1</code>",
            sort_by=sort_by,
        )
    )

    print("Running policy sweep...")
    pol_df = run_policy_sweep()
    sections.append(
        format_table(
            pol_df,
            "Policy Leaderboard",
            "Fixed forecaster: <code>naive</code> &nbsp;|&nbsp; Fixed demand: <code>semiconductor_ar1</code>",
            sort_by=sort_by,
        )
    )

    print("Running demand generator sweep...")
    dem_df = run_demand_sweep()
    sections.append(
        format_table(
            dem_df,
            "Demand Generator Leaderboard",
            "Fixed policy: <code>order_up_to</code> &nbsp;|&nbsp; Fixed forecaster: <code>naive</code>",
            sort_by=sort_by,
        )
    )

    # Footer
    sections.append("""\
---

<p style="font-size: 0.8em; color: #999; margin-top: 2em;">
Auto-generated by <code>benchmarks/run_leaderboard.py</code>.
To reproduce: <code>python benchmarks/run_leaderboard.py</code>
</p>
""")

    sections.append(SORTABLE_JS)

    md = "\n".join(sections)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)

    elapsed = time.perf_counter() - t_start
    print(f"\nLeaderboard written to {output_path} ({elapsed:.1f}s)")


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
