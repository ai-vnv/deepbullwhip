# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-04-09

### Added

- Modular supply chain simulation framework with ABC-based component injection
- `SemiconductorDemandGenerator` — AR(1) + seasonal + structural shock demand model
- `OrderUpToPolicy` — base-stock ordering policy with configurable service level
- `NewsvendorCost` — holding/backorder cost function with per-echelon parameters
- `SerialSupplyChain` — K-echelon serial chain with `EchelonConfig` and `from_config()` factory
- `VectorizedSupplyChain` — matrix-based `(N, K, T)` engine for Monte Carlo batching (~100x speedup)
- `generate_batch()` — parallel demand path generation for N paths
- 10 publication-grade diagnostic plot functions (KFUPM AI V&V Lab color palette)
- Network diagram and geographic map visualizations
- KFUPM petrochemical supply chain example (Saudi context)
- Forecast sensitivity measurement (`compute_sensitivity`)
- Bullwhip metrics: ratio, fill rate, cumulative bullwhip, theoretical lower bound
- 4 tutorial notebooks: full API guide, cost simulation, bullwhip confirmation, policy comparison
- CLI visualization script (`scripts/visualize.py`)
- 117 unit tests with 99% code coverage
- MkDocs documentation with Material theme
- GitHub Actions CI (multi-OS, Python 3.10-3.13), Codecov, auto-docs, PyPI publish

[0.1.0]: https://github.com/ai-vnv/deepbullwhip/releases/tag/v0.1.0
