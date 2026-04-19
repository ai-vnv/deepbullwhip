"""
End-to-End Newsvendor policy.

Implements the integrated forecast-and-optimize neural-network policy
of Oroojlooyjadid, Snyder & Takáč (2020), "Applying Deep Learning to the
Newsvendor Problem," IISE Transactions, 52(4), 444--463.

Reference code: https://github.com/oroojlooy/newsvendor

The original network predicts the order quantity directly by minimising
the newsvendor loss

    L(y, ŷ) = c_p · max(y - ŷ, 0) + c_h · max(ŷ - y, 0)

rather than a point demand forecast. This wrapper exposes the trained
network to deepbullwhip by treating the critical fractile b / (b + h)
as a service-level proxy. When no checkpoint is provided, the class
falls back to a *data-driven quantile* order: the sample quantile of
the rolling history at level b / (b + h), computed from an internally
estimated h and b using the ``service_level`` (b / (b + h) = service
level).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepbullwhip.policy.base import OrderingPolicy
from deepbullwhip.registry import register


@dataclass
class E2ENewsvendorConfig:
    checkpoint: str | None = None
    feature_window: int = 10
    hidden_sizes: tuple = (128, 64)
    device: str = "cpu"


@register("policy", "e2e_newsvendor")
class E2ENewsvendorPolicy(OrderingPolicy):
    """End-to-End Newsvendor policy (Oroojlooyjadid et al., 2020)."""

    def __init__(
        self,
        lead_time: int = 2,
        service_level: float = 0.95,
        *,
        checkpoint: str | None = None,
        feature_window: int = 10,
        hidden_sizes: tuple = (128, 64),
        device: str = "cpu",
    ) -> None:
        self.lead_time = int(lead_time)
        self.service_level = float(service_level)
        self.config = E2ENewsvendorConfig(
            checkpoint=checkpoint,
            feature_window=feature_window,
            hidden_sizes=hidden_sizes,
            device=device,
        )

        self._hist: List[float] = []
        self._torch = None
        self._device = None
        self._net = None
        if checkpoint is not None:
            self._load_model()

    # ------------------------------------------------------------------ API
    def compute_order(
        self,
        inventory_position: float,
        forecast_mean: float,
        forecast_std: float,
    ) -> float:
        # Keep a rolling feature window for both the NN and the fallback.
        self._hist.append(float(forecast_mean))
        if len(self._hist) > 10 * self.config.feature_window:
            self._hist = self._hist[-10 * self.config.feature_window:]

        if self._net is None:
            # Data-driven quantile fallback.
            q_level = float(self.service_level)
            window = self._hist[-self.config.feature_window:]
            if len(window) < 2:
                target = forecast_mean * (self.lead_time + 1)
            else:
                q = float(np.quantile(window, q_level))
                # Cover lead-time demand.
                target = q * (self.lead_time + 1)
            return float(max(0.0, target - inventory_position))

        torch = self._torch
        w = self.config.feature_window
        hist_w = self._hist[-w:]
        pad = w - len(hist_w)
        if pad > 0:
            hist_w = [0.0] * pad + hist_w
        feat = np.asarray(hist_w + [forecast_std, float(self.lead_time)], dtype=np.float32)
        feat_t = torch.as_tensor(feat, device=self._device).unsqueeze(0)
        with torch.no_grad():
            q_hat = float(self._net(feat_t).squeeze().cpu().numpy())
        target = q_hat * (self.lead_time + 1)
        return float(max(0.0, target - inventory_position))

    def reset(self) -> None:
        self._hist = []

    # -------------------------------------------------------------- internals
    def _load_model(self) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("E2ENewsvendorPolicy(checkpoint=...) requires torch.") from exc
        self._torch = torch
        self._device = torch.device(self.config.device)
        in_dim = self.config.feature_window + 2
        layers = []
        prev = in_dim
        for h in self.config.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Softplus()]  # non-negative output
        self._net = nn.Sequential(*layers).to(self._device)
        state_dict = torch.load(self.config.checkpoint, map_location=self._device)
        self._net.load_state_dict(state_dict)
        self._net.eval()
