# Getting Started

## Installation

```bash
pip install deepbullwhip
```

### From source (development)

```bash
git clone https://github.com/ai-vnv/deepbullwhip.git
cd deepbullwhip
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,docs]"
```

## Core Concepts

DeepBullwhip models a **serial supply chain** where each echelon:

1. **Receives** orders from its pipeline (after lead time delay)
2. **Computes** an order quantity using its **ordering policy**
3. **Satisfies** demand from the downstream echelon
4. **Incurs** cost via its **cost function**

### Swappable Components

| Component | ABC | Default |
|-----------|-----|---------|
| Demand model | `DemandGenerator` | `SemiconductorDemandGenerator` |
| Ordering policy | `OrderingPolicy` | `OrderUpToPolicy` |
| Cost function | `CostFunction` | `NewsvendorCost` |

### Default 4-Echelon Configuration

| Echelon | Role | Lead Time | h | b |
|---------|------|-----------|---|---|
| E1 | Distributor | 2 wk | 0.15 | 0.60 |
| E2 | OSAT | 4 wk | 0.12 | 0.50 |
| E3 | Foundry | 12 wk | 0.08 | 0.40 |
| E4 | Supplier | 8 wk | 0.05 | 0.30 |

## Running Tests

```bash
pytest tests/ -v --cov=deepbullwhip
```

## Generating Figures

```bash
python scripts/visualize.py --save --outdir figures
```
