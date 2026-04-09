# DeepBullwhip

[![CI](https://github.com/ai-vnv/deepbullwhip/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-vnv/deepbullwhip/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ai-vnv/deepbullwhip/graph/badge.svg)](https://codecov.io/gh/ai-vnv/deepbullwhip)
[![Docs](https://img.shields.io/badge/docs-ai--vnv.github.io%2Fdeepbullwhip-006747)](https://ai-vnv.github.io/deepbullwhip)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/ai-vnv/deepbullwhip)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-orange)](https://github.com/ai-vnv/deepbullwhip/releases)

**Multi-tier supply chain bullwhip effect simulator with modular demand models, ordering policies, and cost functions.**

Maintained by the [AI Verification & Validation (AI V&V) Lab](https://ai-vnv.kfupm.io) at King Fahd University of Petroleum & Minerals (KFUPM).

---

## Overview

DeepBullwhip provides a configurable simulation framework for studying the
[bullwhip effect](https://en.wikipedia.org/wiki/Bullwhip_effect) in serial
supply chains. It is designed for researchers and practitioners who need to:

- Simulate multi-echelon supply chains under different demand patterns
- Compare ordering policies (Order-Up-To, custom policies) and cost structures
- Quantify bullwhip amplification, fill rates, and total supply chain costs
- Generate publication-grade diagnostic visualizations
- Run Monte Carlo experiments to study forecast-accuracy vs. robustness tradeoffs

The package is extracted from a computational study on the accuracy–robustness
tradeoff in ML-driven semiconductor supply chains (see `simulation.ipynb`).

## Features

| Component | Description |
|-----------|-------------|
| **Demand generators** | Pluggable via `DemandGenerator` ABC. Default: AR(1) + seasonal + structural shock, calibrated to WSTS semiconductor data |
| **Ordering policies** | Pluggable via `OrderingPolicy` ABC. Default: Order-Up-To (OUT / base-stock) with configurable service level |
| **Cost functions** | Pluggable via `CostFunction` ABC. Default: Newsvendor (holding + backorder) with per-echelon h and b |
| **Supply chain** | `SerialSupplyChain` supporting arbitrary K-echelon serial topologies via `EchelonConfig` |
| **Diagnostics** | 10 publication-grade plot functions + network diagram + geographic map visualization |
| **Metrics** | Bullwhip ratio, fill rate, cumulative bullwhip, theoretical lower bounds |
| **Vectorized engine** | `VectorizedSupplyChain` — matrix-based `(N, K, T)` simulation for Monte Carlo batching. **~100x speedup** over serial for N=1000 paths |

## Installation

```bash
# Clone the repository
git clone https://github.com/ai-vnv/deepbullwhip.git
cd deepbullwhip

# Create virtual environment and install
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Dependencies

- **Core:** numpy, scipy, pandas, matplotlib
- **Dev:** pytest, pytest-cov
- **Optional (ML):** scikit-learn, torch

## Quick Start

```python
import numpy as np
from deepbullwhip import (
    SemiconductorDemandGenerator,
    SerialSupplyChain,
)

# 1. Generate demand (156 weeks, with shock at week 104)
gen = SemiconductorDemandGenerator()
demand = gen.generate(T=156, seed=42)

# 2. Simulate the default 4-echelon semiconductor supply chain
chain = SerialSupplyChain()
forecasts_mean = np.full_like(demand, demand.mean())
forecasts_std = np.full_like(demand, demand.std())
result = chain.simulate(demand, forecasts_mean, forecasts_std)

# 3. Inspect results
for k, er in enumerate(result.echelon_results):
    print(f"E{k+1}: {er.name:12s}  BW={er.bullwhip_ratio:.2f}  "
          f"FR={er.fill_rate:.0%}  Cost={er.total_cost:,.0f}")
```

## Default Supply Chain Configuration

| Echelon | Role | Lead Time | h (holding) | b (backorder) |
|---------|------|-----------|-------------|---------------|
| E1 | Distributor / OEM | 2 weeks | 0.15 | 0.60 |
| E2 | Assembly & Test (OSAT) | 4 weeks | 0.12 | 0.50 |
| E3 | Foundry / Fab | 12 weeks | 0.08 | 0.40 |
| E4 | Wafer / Material Supplier | 8 weeks | 0.05 | 0.30 |

## Vectorized Monte Carlo Simulation

For large-scale experiments, use the matrix-based engine that processes
N demand paths simultaneously via NumPy broadcasting:

```python
from deepbullwhip import SemiconductorDemandGenerator, VectorizedSupplyChain

gen = SemiconductorDemandGenerator()
demand = gen.generate_batch(T=156, n_paths=1000, seed=42)  # (1000, 156)

vchain = VectorizedSupplyChain()
fm = np.full_like(demand, demand.mean())
fs = np.full_like(demand, demand.std())
result = vchain.simulate(demand, fm, fs)

# Average metrics across all 1000 paths
print(result.mean_metrics())

# Extract a single path as standard SimulationResult
sr = result.to_simulation_result(path_index=0)
```

**Benchmark (N=1000, T=156, K=4):**

| Engine | Time | Speedup |
|--------|------|---------|
| Serial (`SerialSupplyChain`) | 3.9s | 1x |
| Vectorized (`VectorizedSupplyChain`) | 0.04s | **~100x** |

The vectorized engine uses:
- Pre-allocated `(N, K, T)` order/inventory/cost matrices
- Circular buffer pipeline with O(1) indexing (vs O(L) list.pop)
- Fully vectorized OUT policy and newsvendor cost across N paths and K echelons per time step
- Batch demand generation via `generate_batch()` with `(N, T)` noise matrix

## Customization

### Custom echelon configuration

```python
from deepbullwhip import EchelonConfig, SerialSupplyChain

configs = [
    EchelonConfig("Retailer", lead_time=1, holding_cost=0.20, backorder_cost=0.80),
    EchelonConfig("Manufacturer", lead_time=6, holding_cost=0.10, backorder_cost=0.40),
]
chain = SerialSupplyChain.from_config(configs)
```

### Custom ordering policy

```python
from deepbullwhip.policy.base import OrderingPolicy

class MyPolicy(OrderingPolicy):
    def compute_order(self, inventory_position, forecast_mean, forecast_std):
        # Your logic here
        return max(0.0, forecast_mean - inventory_position)
```

### Custom cost function

```python
from deepbullwhip.cost.base import CostFunction

class MyCost(CostFunction):
    def compute(self, inventory):
        # Your logic here
        return abs(inventory) * 0.1
```

## Visualization

### Diagnostic plots

All plot functions return `matplotlib.figure.Figure` objects and support
`width="single"` (3.5") or `width="double"` (7.0") for journal formatting.
Colors use the KFUPM AI V&V Lab palette.

```python
from deepbullwhip.diagnostics.plots import (
    plot_demand_trajectory,
    plot_order_quantities,
    plot_inventory_levels,
    plot_inventory_position,
    plot_order_streams,
    plot_cost_timeseries,
    plot_cost_decomposition,
    plot_bullwhip_amplification,
    plot_summary_dashboard,
    plot_echelon_detail,
)

fig = plot_summary_dashboard(demand, result)
fig.savefig("dashboard.pdf", dpi=300)
```

### Network and geographic visualization

```python
from deepbullwhip.diagnostics.network import (
    kfupm_petrochemical_network,
    plot_network_diagram,
    plot_supply_chain_map,
)

network = kfupm_petrochemical_network()
fig = plot_network_diagram(network, sim_result=result)
fig = plot_supply_chain_map(network, sim_result=result)
```

### Batch figure generation

```bash
python scripts/visualize.py --save --outdir figures --dpi 600
```

## Project Structure

```
deepbullwhip/
├── __init__.py                 # Public API re-exports
├── _types.py                   # TimeSeries, EchelonResult, SimulationResult
├── sensitivity.py              # Forecast sensitivity (lambda_f)
├── demand/
│   ├── base.py                 # DemandGenerator ABC
│   └── semiconductor.py        # AR(1) + seasonal + shock
├── policy/
│   ├── base.py                 # OrderingPolicy ABC
│   └── order_up_to.py          # Order-Up-To (OUT) policy
├── cost/
│   ├── base.py                 # CostFunction ABC
│   └── newsvendor.py           # Newsvendor h/b cost
├── chain/
│   ├── config.py               # EchelonConfig + defaults
│   ├── echelon.py              # SupplyChainEchelon
│   ├── serial.py               # SerialSupplyChain
│   └── vectorized.py           # VectorizedSupplyChain (N,K,T) matrix engine
└── diagnostics/
    ├── metrics.py              # Bullwhip ratio, fill rate, etc.
    ├── plots.py                # 10 publication-grade plot functions
    └── network.py              # Network diagram + geographic map

tests/                          # 117 unit tests, 99% coverage
notebooks/
├── tutorial.ipynb              # Full API walkthrough
├── 01_supply_chain_cost.ipynb  # Cost simulation & service level tradeoffs
├── 02_bullwhip_effect.ipynb    # Bullwhip confirmation & Monte Carlo validation
└── 03_custom_policies.ipynb    # Comparing ordering policies (OUT, fixed, smoothed)
scripts/visualize.py            # CLI figure generation
simulation.ipynb                # Original research notebook
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=deepbullwhip --cov-report=term-missing
```

Current: **117 tests, 99% code coverage**.

## Tutorial

See [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) for a step-by-step
guide covering demand generation, chain configuration, simulation, visualization,
and custom policy implementation.

## Citation

If you use DeepBullwhip in your research, please cite:

```bibtex
@software{deepbullwhip,
  title  = {DeepBullwhip: Multi-Tier Supply Chain Bullwhip Effect Simulator},
  author = {Arief, Mansur M.},
  url    = {https://github.com/ai-vnv/deepbullwhip},
  year   = {2025}
}
```

## Documentation

Full API documentation is available at [ai-vnv.github.io/deepbullwhip](https://ai-vnv.github.io/deepbullwhip).

## License

MIT License. See [LICENSE](LICENSE) for details.

Developed and maintained by the [AI V&V Lab](https://ai-vnv.kfupm.io) at KFUPM.
