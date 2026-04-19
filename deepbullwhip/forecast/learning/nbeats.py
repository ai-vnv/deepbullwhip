"""
N-BEATS forecaster.

Oreshkin, Carpov, Chapados & Bengio (2020), "N-BEATS: Neural basis
expansion analysis for interpretable time series forecasting," ICLR.

Reference code: https://github.com/philipperemy/n-beats (keras/pytorch)
                neuralforecast / pytorch-forecasting (production)

Fallback
--------
When the ``nbeats`` optional dep is missing, a closed-form
*polynomial + seasonality basis regression* is fitted on the rolling
window, mimicking the interpretable N-BEATS block's trend-and-
seasonality projection. This keeps the forecaster usable inside
BenchmarkRunner on systems without torch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.registry import register


@register("forecaster", "nbeats")
class NBEATSForecaster(Forecaster):
    """N-BEATS forecaster (Oreshkin et al., 2020)."""

    def __init__(
        self,
        *,
        model=None,
        context_length: int = 52,
        trend_degree: int = 3,
        season_period: int = 52,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self.context_length = int(context_length)
        self.trend_degree = int(trend_degree)
        self.season_period = int(season_period)
        self.device = device

    def forecast(
        self, demand_history: np.ndarray, steps_ahead: int = 1
    ) -> tuple[float, float]:
        h = np.asarray(demand_history, dtype=float)
        if h.size < 3:
            mean = float(np.mean(h)) if h.size else 0.0
            std = float(np.std(h)) if h.size > 1 else 0.0
            return mean, std

        if self._model is not None:
            try:
                return self._neural_forecast(h, steps_ahead)
            except Exception:  # pragma: no cover - graceful degrade
                pass

        return self._basis_forecast(h, steps_ahead)

    # ------------------------------------------------------------------ fallback
    def _basis_forecast(self, h: np.ndarray, steps_ahead: int) -> tuple[float, float]:
        """Interpretable N-BEATS-style trend + seasonality projection."""
        ctx = h[-self.context_length:]
        T = ctx.size
        t = np.arange(T, dtype=float) / max(T - 1, 1)

        # Trend basis: polynomial of degree <= trend_degree
        trend_basis = np.stack([t ** k for k in range(self.trend_degree + 1)], axis=1)

        # Seasonality basis: Fourier with self.season_period
        n_harmonics = min(4, max(1, T // 4))
        if self.season_period > 1 and T >= 2 * self.season_period:
            freqs = np.arange(1, n_harmonics + 1) * (2 * np.pi / self.season_period)
            idx = np.arange(T, dtype=float)
            sin_basis = np.stack([np.sin(f * idx) for f in freqs], axis=1)
            cos_basis = np.stack([np.cos(f * idx) for f in freqs], axis=1)
            basis = np.concatenate([trend_basis, sin_basis, cos_basis], axis=1)
        else:
            basis = trend_basis

        coefs, residuals, _, _ = np.linalg.lstsq(basis, ctx, rcond=None)

        # Project one step into the future.
        future_t = np.array([T / max(T - 1, 1)])
        future_trend = np.stack([future_t ** k for k in range(self.trend_degree + 1)],
                                axis=1)
        if basis.shape[1] > trend_basis.shape[1]:
            f_idx = np.array([float(T + steps_ahead - 1)])
            future_sin = np.stack([np.sin(f * f_idx) for f in freqs], axis=1)
            future_cos = np.stack([np.cos(f * f_idx) for f in freqs], axis=1)
            future_basis = np.concatenate([future_trend, future_sin, future_cos], axis=1)
        else:
            future_basis = future_trend
        proj = np.asarray(future_basis @ coefs).ravel()
        mean = float(proj[0])
        # Residual-based std
        preds = basis @ coefs
        resid = ctx - preds
        std = float(np.std(resid)) if resid.size > 1 else 0.0
        # Guard against negative point forecast
        mean = max(mean, 0.0)
        return mean, std

    def _neural_forecast(self, h: np.ndarray, steps_ahead: int) -> tuple[float, float]:
        """Hook for a trained torch N-BEATS model (lazy)."""
        import torch
        ctx = h[-self.context_length:]
        x = torch.as_tensor(ctx, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._model(x).squeeze().cpu().numpy()
        step = out[min(steps_ahead - 1, len(np.atleast_1d(out)) - 1)]
        resid = ctx - float(np.mean(ctx))
        return float(max(step, 0.0)), float(np.std(resid))
