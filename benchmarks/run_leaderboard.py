"""
Generate the benchmark leaderboard: one combined interactive HTML table.

Usage:
    python benchmarks/run_leaderboard.py [--output docs/LEADERBOARD.md]
    python benchmarks/run_leaderboard.py --demands all --policies all \\
        --forecasters all --default-metrics BWR,CUM_BWR,FILL_RATE,TC

Writes ``leaderboard.html`` next to the markdown output (same directory)
and a short ``LEADERBOARD.md`` that embeds it. The HTML page uses checklists
to filter demand, policy, forecaster rows and metric columns; defaults match
BenchmarkRunner's default metrics for column visibility.
"""

from __future__ import annotations

import argparse
import json
import html as html_module
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Import all modules so @register decorators fire
import deepbullwhip.demand  # noqa: F401
import deepbullwhip.policy  # noqa: F401
import deepbullwhip.forecast  # noqa: F401
import deepbullwhip.metrics  # noqa: F401
import deepbullwhip.ext  # noqa: F401
from deepbullwhip.benchmark import BenchmarkRunner
from deepbullwhip.registry import get_class, list_registered

# ── Protocol defaults ──────────────────────────────────────────────────
SEED = 966
T = 156
N = 1000
CHAIN = "semiconductor_4tier"

# BenchmarkRunner.run(..., metrics=None) uses these four:
DEFAULT_METRICS: list[str] = ["BWR", "CUM_BWR", "FILL_RATE", "TC"]

METRIC_LABELS: dict[str, str] = {
    "BWR": "BWR",
    "CUM_BWR": "Cum. BWR",
    "FILL_RATE": "Fill rate",
    "TC": "Total cost",
    "NSAmp": "NS amp.",
    "RFU": "RFU",
    "OSR": "OSR",
    "PeakBWR": "Peak BWR",
    "ExpectedShortfall": "Exp. shortfall",
    "InventoryTurnover": "Inv. turnover",
    "DampingRatio": "Damping",
}


def _benchmark_metric_names() -> list[str]:
    """Metrics usable with BenchmarkRunner (have ``compute``)."""
    return sorted(
        n
        for n in list_registered("metric")
        if callable(getattr(get_class("metric", n), "compute", None))
    )


def _parse_dim(arg: str | None, universe: list[str], label: str) -> list[str]:
    if arg is None or str(arg).strip().lower() == "all":
        return list(universe)
    out = [x.strip() for x in str(arg).split(",") if x.strip()]
    bad = [x for x in out if x not in universe]
    if bad:
        raise SystemExit(f"Unknown {label} name(s): {bad}. Valid: {universe}")
    return out


def _parse_demands(arg: str | None) -> list[str]:
    reg = [d for d in list_registered("demand") if d != "replay"]
    return _parse_dim(arg, reg, "demand")


def _sanitize_json_value(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    return v


def build_combined_frame(
    demands: list[str],
    policies: list[str],
    forecasters: list[str],
    metrics: list[str],
    *,
    t: int,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Wide table: demand, policy, forecaster, <metric columns> at upstream echelon."""
    parts: list[pd.DataFrame] = []
    for d in demands:
        runner = BenchmarkRunner(CHAIN, d, T=t, N=n, seed=seed)
        long_df = runner.run(
            policies=policies,
            forecasters=forecasters,
            metrics=metrics,
        )
        top = long_df["echelon"].max()
        e_top = long_df[long_df["echelon"] == top]
        wide = e_top.pivot_table(
            index=["policy", "forecaster"],
            columns="metric",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide.insert(0, "demand", d)
        parts.append(wide)
    out = pd.concat(parts, ignore_index=True)
    meta = ["demand", "policy", "forecaster"]
    mcols = sorted([c for c in out.columns if c not in meta])
    return out[meta + mcols]


def render_interactive_html(
    df: pd.DataFrame,
    *,
    chain: str,
    t: int,
    n: int,
    seed: int,
    default_demands: list[str],
    default_metrics: list[str],
) -> str:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = {k: _sanitize_json_value(row[k]) for k in df.columns}
        records.append(rec)

    demands = sorted(df["demand"].unique().tolist())
    policies = sorted(df["policy"].unique().tolist())
    forecasters = sorted(df["forecaster"].unique().tolist())
    metric_cols = [c for c in df.columns if c not in ("demand", "policy", "forecaster")]

    payload = {
        "rows": records,
        "dims": {"demands": demands, "policies": policies, "forecasters": forecasters},
        "metrics": metric_cols,
        "labels": {m: METRIC_LABELS.get(m, m) for m in metric_cols},
        "defaults": {
            "demands": [d for d in default_demands if d in demands],
            "policies": policies,
            "forecasters": forecasters,
            "metrics": [m for m in default_metrics if m in metric_cols],
        },
        "protocol": {
            "chain": chain,
            "T": t,
            "N": n,
            "seed": seed,
            "echelon": "E4 (max upstream in table)",
        },
        "initialSort": ("TC" if "TC" in metric_cols else (metric_cols[0] if metric_cols else "demand")),
    }
    json_txt = json.dumps(payload, separators=(",", ":"))

    # Minimal single-file UI: checklists + one sortable table
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Benchmark leaderboard</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:12px;font-size:13px;color:#1a1a1a;line-height:1.35}}
h1{{font-size:1.1rem;margin:0 0 6px;font-weight:600}}
.proto{{font-family:ui-monospace,monospace;font-size:11px;color:#555;margin-bottom:10px}}
fieldset{{border:1px solid #ccc;border-radius:4px;margin:0 0 8px;padding:6px 8px;background:#fafafa}}
legend{{font-size:11px;font-weight:600;padding:0 4px}}
.chkrow{{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;margin-top:4px}}
.chkrow label{{font-size:12px;white-space:nowrap;cursor:pointer;display:inline-flex;gap:4px;align-items:center}}
table{{border-collapse:collapse;width:100%;margin-top:8px;font-size:12px}}
th,td{{border:1px solid #ddd;padding:4px 7px}}
th{{background:#eee;text-align:left;cursor:pointer;user-select:none;font-weight:600}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
th.num,td.num{{text-align:right}}
tr:nth-child(even) td{{background:#fcfcfc}}
.hidden{{display:none}}
.toolbar{{margin:6px 0;font-size:11px}}
.toolbar button{{font-size:11px;margin-right:6px;padding:2px 8px;cursor:pointer;border:1px solid #bbb;border-radius:3px;background:#fff}}
</style>
</head>
<body>
<h1>Benchmark leaderboard</h1>
<div class="proto" id="proto"></div>

<fieldset><legend>Demand</legend><div class="chkrow" id="demands"></div></fieldset>
<fieldset><legend>Policy</legend><div class="chkrow" id="policies"></div></fieldset>
<fieldset><legend>Forecaster</legend><div class="chkrow" id="forecasters"></div></fieldset>
<fieldset><legend>Metrics</legend><div class="chkrow" id="metrics"></div></fieldset>

<div class="toolbar">
<button type="button" id="btn-all">All filters on</button>
<button type="button" id="btn-def">Defaults (metrics)</button>
</div>

<table id="tbl">
<thead><tr id="headrow"></tr></thead>
<tbody id="body"></tbody>
</table>

<script id="lb-data" type="application/json">{json_txt}</script>
<script>
(function () {{
  const DATA = JSON.parse(document.getElementById("lb-data").textContent);
  const proto = document.getElementById("proto");
  const p = DATA.protocol;
  proto.textContent = "chain=" + p.chain + " | T=" + p.T + " | N=" + p.N + " | seed=" + p.seed + " | " + p.echelon;

  const state = {{ d: {{}}, p: {{}}, f: {{}}, m: {{}} }};
  let sortKey = DATA.initialSort || "demand";
  let sortAsc = true;

  function mkGroup(containerId, items, key, defList) {{
    const el = document.getElementById(containerId);
    const defSet = new Set(defList && defList.length ? defList : items);
    items.forEach(function (name) {{
      const id = key + "_" + name.replace(/[^a-zA-Z0-9_]/g, "_");
      const lab = document.createElement("label");
      const inp = document.createElement("input");
      inp.type = "checkbox";
      inp.id = id;
      inp.checked = defSet.has(name);
      inp.addEventListener("change", render);
      lab.appendChild(inp);
      lab.appendChild(document.createTextNode(name));
      el.appendChild(lab);
      state[key][name] = inp;
    }});
  }}

  mkGroup("demands", DATA.dims.demands, "d", DATA.defaults.demands);
  mkGroup("policies", DATA.dims.policies, "p", DATA.defaults.policies);
  mkGroup("forecasters", DATA.dims.forecasters, "f", DATA.defaults.forecasters);
  mkGroup("metrics", DATA.metrics, "m", DATA.defaults.metrics);

  function checkedList(key) {{
    return Object.keys(state[key]).filter(function (k) {{ return state[key][k].checked; }});
  }}

  function fmtCell(metric, val) {{
    if (val === null || val === undefined) return "—";
    if (val === "inf") return "∞";
    if (val === "-inf") return "−∞";
    if (metric === "FILL_RATE" && typeof val === "number")
      return (100 * val).toFixed(1) + "%";
    if (typeof val === "number")
      return val.toLocaleString(undefined, {{ maximumFractionDigits: 1 }});
    return String(val);
  }}

  function parseSort(metric, cellText) {{
    if (metric === "demand" || metric === "policy" || metric === "forecaster")
      return cellText;
    if (cellText === "—" || cellText === "∞" || cellText === "−∞") return NaN;
    return parseFloat(String(cellText).replace(/,/g, "").replace("%", ""));
  }}

  function cmpRows(a, b, key, asc) {{
    let v;
    if (key === "demand" || key === "policy" || key === "forecaster") {{
      v = String(a[key]).localeCompare(String(b[key]));
    }} else {{
      const na = parseSort(key, fmtCell(key, a[key]));
      const nb = parseSort(key, fmtCell(key, b[key]));
      v = (isNaN(na) ? (isNaN(nb) ? 0 : 1) : (isNaN(nb) ? -1 : na - nb));
    }}
    return asc ? v : -v;
  }}

  function render() {{
    const ds = new Set(checkedList("d"));
    const ps = new Set(checkedList("p"));
    const fs = new Set(checkedList("f"));
    const ms = checkedList("m");

    const head = document.getElementById("headrow");
    head.innerHTML = "";
    const cols = ["demand", "policy", "forecaster"].concat(ms);
    cols.forEach(function (col) {{
      const th = document.createElement("th");
      th.textContent = (col === "demand" || col === "policy" || col === "forecaster")
        ? col
        : (DATA.labels[col] || col);
      if (col !== "demand" && col !== "policy" && col !== "forecaster") th.className = "num";
      th.dataset.sortKey = col;
      if (col === sortKey) th.dataset.dir = sortAsc ? "asc" : "desc";
      th.addEventListener("click", function () {{
        if (sortKey === col) sortAsc = !sortAsc;
        else {{ sortKey = col; sortAsc = true; }}
        render();
      }});
      head.appendChild(th);
    }});

    let rows = DATA.rows.filter(function (r) {{
      return ds.has(r.demand) && ps.has(r.policy) && fs.has(r.forecaster);
    }});
    rows = rows.slice().sort(function (a, b) {{ return cmpRows(a, b, sortKey, sortAsc); }});

    const body = document.getElementById("body");
    body.innerHTML = "";
    rows.forEach(function (r) {{
      const tr = document.createElement("tr");
      ["demand", "policy", "forecaster"].forEach(function (c) {{
        const td = document.createElement("td");
        td.textContent = r[c];
        tr.appendChild(td);
      }});
      ms.forEach(function (m) {{
        const td = document.createElement("td");
        td.className = "num";
        td.textContent = fmtCell(m, r[m]);
        tr.appendChild(td);
      }});
      body.appendChild(tr);
    }});
  }}

  document.getElementById("btn-all").addEventListener("click", function () {{
    ["d", "p", "f", "m"].forEach(function (k) {{
      Object.keys(state[k]).forEach(function (name) {{ state[k][name].checked = true; }});
    }});
    render();
  }});
  document.getElementById("btn-def").addEventListener("click", function () {{
    Object.keys(state.d).forEach(function (n) {{
      state.d[n].checked = DATA.defaults.demands.indexOf(n) >= 0;
    }});
    Object.keys(state.p).forEach(function (n) {{ state.p[n].checked = true; }});
    Object.keys(state.f).forEach(function (n) {{ state.f[n].checked = true; }});
    Object.keys(state.m).forEach(function (n) {{
      state.m[n].checked = DATA.defaults.metrics.indexOf(n) >= 0;
    }});
    sortKey = DATA.initialSort || "demand";
    sortAsc = true;
    render();
  }});

  render();
}})();
</script>
</body>
</html>
"""


def write_markdown_stub(md_path: Path, html_name: str) -> None:
    """Short MkDocs page that embeds the interactive HTML."""
    body = f"""# Benchmark Leaderboard

One combined table: filter **demand**, **policy**, and **forecaster** rows and **metric** columns with the checklists. Default visible metrics match `BenchmarkRunner` defaults (`BWR`, `CUM_BWR`, `FILL_RATE`, `TC`). Regenerate with `benchmarks/run_leaderboard.py` (see `--help`).

<iframe src="{html_module.escape(html_name)}" title="Benchmark leaderboard" width="100%" height="860" style="border:1px solid #ddd;border-radius:4px"></iframe>

If the frame does not load, open [{html_name}]({html_name}) directly.
"""
    md_path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate combined interactive benchmark leaderboard (HTML + short MD).",
    )
    parser.add_argument(
        "--output",
        default="docs/LEADERBOARD.md",
        help="Markdown output path; leaderboard.html is written alongside it.",
    )
    parser.add_argument("--T", type=int, default=T, help=f"periods (default {T})")
    parser.add_argument("--N", type=int, default=N, help=f"Monte Carlo paths (default {N})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"seed (default {SEED})")
    parser.add_argument(
        "--demands",
        default="semiconductor_ar1",
        help="Comma-separated demand names, or 'all' (excludes replay). Default: semiconductor_ar1.",
    )
    parser.add_argument(
        "--policies",
        default="all",
        help="Comma-separated policy names or 'all'.",
    )
    parser.add_argument(
        "--forecasters",
        default="all",
        help="Comma-separated forecaster names or 'all'.",
    )
    parser.add_argument(
        "--default-metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics checked on page load (subset of computed metrics).",
    )
    parser.add_argument(
        "--compute-metrics",
        default="all",
        help="Comma-separated metrics to compute and embed, or 'all' (simulation-backed metrics only).",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    out_md = Path(args.output)
    html_name = "leaderboard.html"
    out_html = out_md.with_name(html_name)

    policies = _parse_dim(args.policies, list_registered("policy"), "policy")
    forecasters = _parse_dim(args.forecasters, list_registered("forecaster"), "forecaster")
    demands = _parse_demands(args.demands)
    all_m = _benchmark_metric_names()
    if str(args.compute_metrics).strip().lower() == "all":
        metrics = all_m
    else:
        metrics = [x.strip() for x in str(args.compute_metrics).split(",") if x.strip()]
        bad = [m for m in metrics if m not in all_m]
        if bad:
            raise SystemExit(f"--compute-metrics unknown: {bad}. Valid: {all_m}")
    default_metrics = [
        x.strip()
        for x in str(args.default_metrics).split(",")
        if x.strip()
    ]
    bad_m = [m for m in default_metrics if m not in metrics]
    if bad_m:
        raise SystemExit(
            f"--default-metrics must be subset of computed metrics. Unknown: {bad_m}. "
            f"Computed: {metrics}"
        )

    if len(demands) == 1:
        default_demands = [demands[0]]
    else:
        default_demands = [d for d in demands if d == "semiconductor_ar1"] or [demands[0]]

    print(f"Demands={demands}  policies={len(policies)}  forecasters={len(forecasters)}  metrics={len(metrics)}")
    print(f"Building combined table (T={args.T}, N={args.N})...")
    df = build_combined_frame(
        demands,
        policies,
        forecasters,
        metrics,
        t=args.T,
        n=args.N,
        seed=args.seed,
    )

    html_doc = render_interactive_html(
        df,
        chain=CHAIN,
        t=args.T,
        n=args.N,
        seed=args.seed,
        default_demands=default_demands,
        default_metrics=default_metrics,
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc, encoding="utf-8")
    write_markdown_stub(out_md, html_name)

    elapsed = time.perf_counter() - t_start
    print(f"Wrote {out_html} and {out_md} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
