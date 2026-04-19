# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-04-19

### Added

- **`deepbullwhip.ext`**: optional learning policies (`dqn_beer_game`, `recurrent_ppo`, `dcl`, `e2e_newsvendor`), forecasters (`nbeats`, `tft`, `lightgbm_quantile`, `lstm_multistep`), and metrics (`RFU`, `OSR`, `PeakBWR`, `ExpectedShortfall`, `InventoryTurnover`, `DampingRatio`) with lazy or fallback paths for minimal installs.
- **Optional dependency groups** in `pyproject.toml`: `gbm`, `torch`, `rl`, `neural`, `learning` (plus `all` extended).
- **`scripts/run_learning_leaderboard.py`**: policy×forecaster composite leaderboard CLI.
- **V&V**: extension requirements `REQ-EXT-001`–`REQ-EXT-008` in `vnvspec.yaml`, evidence in `scripts/vnv_assess.py`, and `tests/test_vnvspec.py` (`vnvspec validate` + assessment script). `dev` extras include `vnvspec`, `typer`, and `rich`.

### Changed

- **Benchmark leaderboard** (`benchmarks/run_leaderboard.py`): single combined interactive **`docs/leaderboard.html`** with checklists for demand, policy, forecaster, and metric columns; short **`docs/LEADERBOARD.md`** embeds it. CLI flags for dimensions, `--compute-metrics`, and `--default-metrics` (defaults align with `BenchmarkRunner`).

[0.5.0]: https://github.com/ai-vnv/deepbullwhip/releases/tag/v0.5.0

## [0.2.0] - 2026-04-10

### Added

- **Registry system** (`registry.py`): Decorator-based `@register` / `get` / `list_registered` for all component types
- **New ordering policies**: ProportionalOUTPolicy (POUT), ConstantOrderPolicy, SmoothingOUTPolicy
- **New demand generators**: BeerGameDemandGenerator, ARMADemandGenerator, ReplayDemandGenerator
- **Forecaster module** (`forecast/`): Forecaster ABC with NaiveForecaster, MovingAverageForecaster, ExponentialSmoothingForecaster
- **Metrics module** (`metrics/`): Structured BWR, CumulativeBWR, NSAmp, FillRate, TotalCost, ChenLowerBound with `@register` support
- **Benchmark module** (`benchmark/`): BenchmarkRunner for standardized policy/forecaster comparison with LaTeX, Markdown, and CSV export
- **Datasets module** (`datasets/`): Built-in Beer Game, WSTS semiconductor sample, synthetic AR(1)/ARMA, M5 Walmart loader
- **Perishable cost function**: NewsvendorCost + obsolescence penalty for technology perishability
- **Predefined chain configs**: semiconductor_4tier, beer_game, consumer_2tier
- 3 benchmark notebooks: policy comparison, forecaster comparison, add-your-own-model tutorial
- 102 new tests (219 total)

### Changed

- Version bumped to 0.2.0
- `__init__.py` exports all new classes while maintaining full backward compatibility
- `pyproject.toml` adds `benchmark` optional dependency group

### Backward Compatibility

- All v0.1.0 imports and APIs remain unchanged
- `diagnostics.metrics` module continues to work as before
- `simulate()` interface unchanged (forecasts_mean/std as separate arrays)
- All 117 original tests pass without modification

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

[0.2.0]: https://github.com/ai-vnv/deepbullwhip/releases/tag/v0.2.0
[0.1.0]: https://github.com/ai-vnv/deepbullwhip/releases/tag/v0.1.0
