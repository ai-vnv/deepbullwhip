# DeepBullwhip

**Multi-tier supply chain bullwhip effect simulator with modular demand models, ordering policies, and cost functions.**

Maintained by the [AI V&V Lab](https://ai-vnv.kfupm.io) at KFUPM.

## Features

- **Modular architecture** — swap demand generators, ordering policies, and cost functions via ABCs
- **Configurable multi-echelon chains** — define any serial supply chain topology
- **Vectorized Monte Carlo engine** — ~100x speedup for batch simulations
- **Publication-grade diagnostics** — 10 plot functions + network/map visualizations
- **99% test coverage** — comprehensive unit tests across all modules

## Installation

```bash
pip install deepbullwhip
```

Or install from source:

```bash
git clone https://github.com/ai-vnv/deepbullwhip.git
cd deepbullwhip
pip install -e ".[dev]"
```

## Quick Example

```python
import numpy as np
from deepbullwhip import SemiconductorDemandGenerator, SerialSupplyChain

gen = SemiconductorDemandGenerator()
demand = gen.generate(T=156, seed=966)

chain = SerialSupplyChain()
fm = np.full_like(demand, demand.mean())
fs = np.full_like(demand, demand.std())
result = chain.simulate(demand, fm, fs)

for er in result.echelon_results:
    print(f"{er.name}: BW={er.bullwhip_ratio:.2f}, FR={er.fill_rate:.0%}")
```
