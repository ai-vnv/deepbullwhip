"""
LightGBM Quantile-Regression forecaster.

Huber, Gossmann & Stuckenschmidt (2019), "A Data-Driven Newsvendor
Problem: From Data to Decision," European Journal of Operational
Research, 278(3), 904--915.

Reference code: the ``lightgbm`` library supports quantile loss natively
(``objective="quantile"``).

The forecaster fits three LightGBM regressors (q10, q50, q90) on a
rolling embedding of the demand series and returns the median as the
point forecast together with a spread-based std estimate. For early
periods where the window is too short to fit a GBM, it falls back to
rolling-mean / rolling-std so BenchmarkRunner always gets valid
numbers.
"""

from __future__ import annotations


import numpy as np

from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.registry import register


@register("forecaster", "lightgbm_quantile")
class LightGBMQuantileForecaster(Forecaster):
    """LightGBM quantile-regression forecaster (Huber et al., 2019)."""

    def __init__(
        self,
        *,
        lags: int = 8,
        n_estimators: int = 100,
        num_leaves: int = 15,
        learning_rate: float = 0.1,
        quantiles: tuple = (0.1, 0.5, 0.9),
        min_train_size: int = 30,
        refit_every: int = 10,
    ) -> None:
        self.lags = int(lags)
        self.n_estimators = int(n_estimators)
        self.num_leaves = int(num_leaves)
        self.learning_rate = float(learning_rate)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.min_train_size = int(min_train_size)
        self.refit_every = int(refit_every)

        self._models = {}  # q -> LGBMRegressor
        self._last_fit_size: int = 0
        self._lgb_available: bool | None = None

    # ------------------------------------------------------------------ API
    def forecast(
        self, demand_history: np.ndarray, steps_ahead: int = 1
    ) -> tuple[float, float]:
        h = np.asarray(demand_history, dtype=float)
        n = h.size

        # Warm-up and lightgbm-missing fallback: rolling stats.
        if n < max(self.lags + 2, self.min_train_size) or not self._have_lgbm():
            mean = float(np.mean(h)) if n else 0.0
            std = float(np.std(h)) if n > 1 else 0.0
            return mean, std

        # Refit on a schedule to amortize training cost.
        if (not self._models) or (n - self._last_fit_size >= self.refit_every):
            self._fit(h)

        x = h[-self.lags:].reshape(1, -1)
        preds = {q: float(m.predict(x)[0]) for q, m in self._models.items()}

        median_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
        mean = max(preds[median_q], 0.0)

        if 0.1 in preds and 0.9 in preds:
            std = max(preds[0.9] - preds[0.1], 1e-6) / (2 * 1.2816)
        else:
            std = float(np.std(h[-max(self.lags, 10):]))
        return float(mean), float(std)

    # -------------------------------------------------------------- internals
    def _have_lgbm(self) -> bool:
        if self._lgb_available is None:
            try:
                import lightgbm  # noqa: F401
                self._lgb_available = True
            except ImportError:
                self._lgb_available = False
        return self._lgb_available

    def _fit(self, h: np.ndarray) -> None:
        import lightgbm as lgb
        n = h.size
        X = np.stack([h[i : i + self.lags] for i in range(n - self.lags)])
        y = h[self.lags:]
        self._models = {}
        for q in self.quantiles:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                verbose=-1,
            )
            m.fit(X, y)
            self._models[q] = m
        self._last_fit_size = n
