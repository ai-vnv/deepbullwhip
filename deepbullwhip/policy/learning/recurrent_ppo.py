"""
Recurrent PPO ordering policy.

Wraps the LSTM-based PPO agent of Rozhkov, Alyamovskaya & Zakhodiakin
(2025), "The beer game bullwhip effect mitigation: a deep reinforcement
learning approach," International Journal of Production Research, 63(18),
6630--6647.

The paper uses sb3_contrib.RecurrentPPO with a reward that blends
newsvendor cost and an order-variance penalty. This class is an
inference-time wrapper: training is delegated to an external script
(``scripts/train_recurrent_ppo.py``) so that the extension package
remains importable without stable-baselines3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from deepbullwhip.policy.base import OrderingPolicy
from deepbullwhip.registry import register


@dataclass
class RecurrentPPOConfig:
    model_path: Optional[str] = None
    state_window: int = 10
    action_scale: float = 1.0
    action_offset: float = 0.0
    device: str = "cpu"


@register("policy", "recurrent_ppo")
class RecurrentPPOPolicy(OrderingPolicy):
    """Recurrent PPO policy (Rozhkov et al., 2025)."""

    def __init__(
        self,
        lead_time: int = 2,
        service_level: float = 0.95,
        *,
        model_path: Optional[str] = None,
        state_window: int = 10,
        action_scale: float = 1.0,
        action_offset: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.lead_time = int(lead_time)
        self.service_level = float(service_level)
        self.config = RecurrentPPOConfig(
            model_path=model_path,
            state_window=state_window,
            action_scale=action_scale,
            action_offset=action_offset,
            device=device,
        )

        self._model = None
        self._lstm_state: Any = None
        self._episode_start = True
        self._buf: List[np.ndarray] = []

        if model_path is not None:
            self._load_model()

    # ------------------------------------------------------------------ API
    def compute_order(
        self,
        inventory_position: float,
        forecast_mean: float,
        forecast_std: float,
    ) -> float:
        if self._model is None:
            # Smoothed OUT fallback: α = 0.5 Proportional OUT.
            S = (self.lead_time + 1) * forecast_mean + 1.645 * forecast_std * np.sqrt(
                self.lead_time + 1
            )
            return float(max(0.0, 0.5 * (S - inventory_position)))

        obs = self._obs(inventory_position, forecast_mean, forecast_std)
        action, self._lstm_state = self._model.predict(
            obs,
            state=self._lstm_state,
            episode_start=np.array([self._episode_start]),
            deterministic=True,
        )
        self._episode_start = False
        a = float(np.asarray(action).ravel()[0])
        return float(max(0.0, a * self.config.action_scale + self.config.action_offset + forecast_mean))

    def reset(self) -> None:
        self._lstm_state = None
        self._episode_start = True
        self._buf = []

    # -------------------------------------------------------------- internals
    def _obs(self, ip: float, fm: float, fs: float) -> np.ndarray:
        feat = np.array([ip, fm, fs, float(self.lead_time)], dtype=np.float32)
        self._buf.append(feat)
        if len(self._buf) > self.config.state_window:
            self._buf = self._buf[-self.config.state_window:]
        pad = self.config.state_window - len(self._buf)
        if pad:
            window = np.stack([np.zeros_like(feat)] * pad + self._buf)
        else:
            window = np.stack(self._buf)
        return window.reshape(-1)

    def _load_model(self) -> None:
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RecurrentPPOPolicy(model_path=...) requires sb3-contrib. "
                "Install with `pip install deepbullwhip[learning]`."
            ) from exc
        self._model = RecurrentPPO.load(self.config.model_path, device=self.config.device)
