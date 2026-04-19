"""
Deep Controlled Learning (DCL) policy.

Wraps the policy network trained via the approximate-policy-iteration
(classification-based) procedure of Temizöz, Imdahl, Dijkman,
Lamghari-Idrissi & van Jaarsveld (2025), "Deep Controlled Learning for
Inventory Control," European Journal of Operational Research, 324(1),
104--117.

Reference implementation: https://github.com/DynaPlex/DynaPlex
                          https://github.com/tarkantemizoz/DynaPlex

Fallback
--------
When no checkpoint is provided this class runs a **capped base-stock**
policy (Xin & Goldberg style), which is the strongest classical
benchmark used in the DCL paper. That gives the policy something
principled to do inside BenchmarkRunner when weights are absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepbullwhip.policy.base import OrderingPolicy
from deepbullwhip.registry import register


@dataclass
class DCLConfig:
    checkpoint: str | None = None
    pipeline_length: int = 12
    action_cap: int = 60
    hidden_sizes: tuple = (128, 128, 128)
    base_stock_cap: float = 80.0
    device: str = "cpu"


@register("policy", "dcl")
class DCLPolicy(OrderingPolicy):
    """Deep Controlled Learning policy (Temizöz et al., 2025)."""

    def __init__(
        self,
        lead_time: int = 2,
        service_level: float = 0.95,
        *,
        checkpoint: str | None = None,
        action_cap: int = 60,
        hidden_sizes: tuple = (128, 128, 128),
        base_stock_cap: float = 80.0,
        device: str = "cpu",
    ) -> None:
        self.lead_time = int(lead_time)
        self.service_level = float(service_level)
        self.config = DCLConfig(
            checkpoint=checkpoint,
            pipeline_length=max(1, int(lead_time)),
            action_cap=action_cap,
            hidden_sizes=hidden_sizes,
            base_stock_cap=base_stock_cap,
            device=device,
        )

        self._pipeline: List[float] = [0.0] * self.config.pipeline_length
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
        if self._net is None:
            target = (self.lead_time + 1) * forecast_mean + 1.645 * forecast_std * np.sqrt(
                self.lead_time + 1
            )
            gap = max(0.0, target - inventory_position)
            return float(min(gap, self.config.base_stock_cap))

        torch = self._torch
        state = np.concatenate([
            np.array([inventory_position], dtype=np.float32),
            np.asarray(self._pipeline, dtype=np.float32),
            np.array([forecast_mean, forecast_std], dtype=np.float32),
        ])
        state_t = torch.as_tensor(state, device=self._device).unsqueeze(0)
        with torch.no_grad():
            logits = self._net(state_t).squeeze(0).cpu().numpy()
        action = int(np.argmax(logits))
        self._pipeline = self._pipeline[1:] + [float(action)]
        return float(action)

    def reset(self) -> None:
        self._pipeline = [0.0] * self.config.pipeline_length

    # -------------------------------------------------------------- internals
    def _load_model(self) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DCLPolicy(checkpoint=...) requires torch."
            ) from exc
        self._torch = torch
        self._device = torch.device(self.config.device)
        in_dim = 1 + self.config.pipeline_length + 2
        layers = []
        prev = in_dim
        for h in self.config.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, self.config.action_cap + 1)]
        self._net = nn.Sequential(*layers).to(self._device)
        state_dict = torch.load(self.config.checkpoint, map_location=self._device)
        self._net.load_state_dict(state_dict)
        self._net.eval()
