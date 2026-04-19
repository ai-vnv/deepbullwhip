"""Optional learning extensions for DeepBullwhip.

Importing this subpackage registers additional policies, forecasters,
and metrics with :mod:`deepbullwhip.registry` so they work with
:class:`~deepbullwhip.benchmark.runner.BenchmarkRunner` by name.

Heavy dependencies (``torch``, ``lightgbm``, ``stable-baselines3``) are
only required when using trained checkpoints or optional code paths;
fallback implementations keep imports usable on minimal installs.

Install extras::

    pip install deepbullwhip[learning]   # all learning extras
    pip install deepbullwhip[gbm]       # LightGBM quantile forecaster
    pip install deepbullwhip[rl]        # RL policies (SB3)
    pip install deepbullwhip[neural]    # pytorch-forecasting hooks
"""

from __future__ import annotations

import deepbullwhip.forecast.learning  # noqa: F401
import deepbullwhip.metrics.damping_ratio  # noqa: F401
import deepbullwhip.metrics.expected_shortfall  # noqa: F401
import deepbullwhip.metrics.inventory_turnover  # noqa: F401
import deepbullwhip.metrics.order_smoothing_ratio  # noqa: F401
import deepbullwhip.metrics.peak_bwr  # noqa: F401
import deepbullwhip.metrics.rfu  # noqa: F401
import deepbullwhip.policy.learning  # noqa: F401

__all__: list[str] = []
