"""
LSTM Multi-step Forecaster.

Motivated by the multi-step-ahead forecasting comparison in recent
bullwhip-effect work (e.g., *Analyzing One-Step and Multi-Step
Forecasting to Mitigate the Bullwhip Effect and Improve Supply Chain
Performance*, cited in surveys of ML-driven safety stock methods).

The class wraps a trained PyTorch LSTM that maps a context window to
a multi-step-ahead forecast trajectory. At each call, it returns the
average of the ``horizon`` predicted steps as the point forecast and
the std of the multi-step predictions as the spread.

When no trained model is supplied, it falls back to an autoregressive
simulation from an AR(2) fit on the rolling window, which preserves
the "multi-step" spirit (the std grows with horizon) and keeps the
forecaster runnable inside BenchmarkRunner without torch.
"""

from __future__ import annotations


import numpy as np

from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.registry import register


@register("forecaster", "lstm_multistep")
class LSTMMultistepForecaster(Forecaster):
    """LSTM multi-step demand forecaster."""

    def __init__(
        self,
        *,
        model=None,
        context_length: int = 26,
        horizon: int = 4,
        hidden_size: int = 64,
        num_layers: int = 1,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self.context_length = int(context_length)
        self.horizon = int(horizon)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.device = device

    def forecast(
        self, demand_history: np.ndarray, steps_ahead: int = 1
    ) -> tuple[float, float]:
        h = np.asarray(demand_history, dtype=float)
        if h.size < 4:
            mean = float(np.mean(h)) if h.size else 0.0
            std = float(np.std(h)) if h.size > 1 else 0.0
            return mean, std

        if self._model is not None:
            try:
                return self._lstm_forecast(h, steps_ahead)
            except Exception:  # pragma: no cover - graceful degrade
                pass

        return self._ar2_multistep_forecast(h, max(steps_ahead, self.horizon))

    # ------------------------------------------------------------------ fallback
    def _ar2_multistep_forecast(
        self, h: np.ndarray, horizon: int
    ) -> tuple[float, float]:
        ctx = h[-self.context_length:]
        n = ctx.size
        if n < 4:
            return float(np.mean(ctx)), float(np.std(ctx))

        # Fit AR(2) by least-squares: x_t = a1 * x_{t-1} + a2 * x_{t-2} + c.
        y = ctx[2:]
        X = np.stack([ctx[1:-1], ctx[:-2], np.ones(n - 2)], axis=1)
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a1, a2, c = coefs
        resid = y - X @ coefs
        sigma = float(np.std(resid)) if resid.size > 1 else 0.0

        # --- Stability check: the AR(2) is stable iff the characteristic
        # roots of 1 - a1 z - a2 z^2 = 0 have magnitude > 1. Equivalently,
        # |a2| < 1, a1 + a2 < 1, and a2 - a1 < 1. If violated, the forward
        # rollout explodes. Fall back to rolling-mean forecasting in that
        # case, which matches the "too little data / non-stationary demand"
        # branch of the original LSTM paper's comparison.
        stable = (abs(a2) < 0.98) and (a1 + a2 < 0.98) and (a2 - a1 < 0.98)
        if not stable:
            mean = float(np.mean(ctx))
            std = float(np.std(ctx))
            return max(mean, 0.0), std

        # Roll forward deterministically; accumulate std across horizon.
        # Clamp each step into [0, 10 * mean(ctx)] as a defensive guard;
        # the clamp is wide enough never to bite on well-behaved data but
        # prevents catastrophic overflow on pathological realisations.
        ctx_mean = float(np.mean(ctx))
        cap = max(10.0 * max(ctx_mean, 1.0), 1.0)

        x_tm1, x_tm2 = ctx[-1], ctx[-2]
        preds = []
        for _ in range(horizon):
            x_t = a1 * x_tm1 + a2 * x_tm2 + c
            x_t = float(np.clip(x_t, -cap, cap))
            preds.append(x_t)
            x_tm2, x_tm1 = x_tm1, x_t

        mean = float(max(np.mean(preds), 0.0))
        # Multi-step std grows roughly as sigma * sqrt(h); cap at context std
        # * 3 so OUT's safety-factor term can't produce astronomical targets
        # when the AR(2) fit has large residual variance on short series.
        ctx_std = float(np.std(ctx))
        std = float(min(sigma * np.sqrt(horizon), 3.0 * max(ctx_std, 1.0)))
        return mean, std

    def _lstm_forecast(self, h: np.ndarray, steps_ahead: int) -> tuple[float, float]:
        """Hook for a trained torch LSTM model (lazy)."""
        import torch
        ctx = h[-self.context_length:]
        x = torch.as_tensor(ctx, dtype=torch.float32).view(1, -1, 1).to(self.device)
        with torch.no_grad():
            out = self._model(x).squeeze().cpu().numpy()
        horizon_preds = np.atleast_1d(out)[: self.horizon]
        mean = float(max(np.mean(horizon_preds), 0.0))
        std = float(np.std(horizon_preds)) if horizon_preds.size > 1 else float(np.std(ctx))
        return mean, std
