"""
DQN policy for the serial supply chain bullwhip.

Implements the Deep Q-Network ordering rule of Oroojlooyjadid, Nazari,
Snyder & Takáč (2022), "A Deep Q-Network for the Beer Game: Deep
Reinforcement Learning for Inventory Optimization," Manufacturing &
Service Operations Management, 24(1), 285--304.

Reference code: https://github.com/OptMLGroup/DeepBeerInventory-RL

Design
------
The original agent selects an integer *deviation* d_k ∈ {-2, -1, 0, 1, 2}
added to the demand signal, producing order = max(0, forecast_mean + d_k).
The Q-network is a plain feed-forward net fed with a windowed state
vector (recent demand, orders, shipments, on-hand).

deepbullwhip's ``compute_order`` ABC only exposes
``(inventory_position, forecast_mean, forecast_std)``, so the policy
maintains its own rolling observation buffer. Observations can be fed
in via ``update_observation`` by a custom driver; if they are never
fed, the window simply stays zero-padded and the network still runs.

If no checkpoint is supplied, the policy falls back to a myopic
order-up-to rule so the benchmark framework can execute it without
any trained weights.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from deepbullwhip.policy.base import OrderingPolicy
from deepbullwhip.registry import register


@dataclass
class DQNBeerGameConfig:
    action_offsets: list[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    state_window: int = 5
    hidden_sizes: tuple = (180, 130, 61, 33)
    epsilon: float = 0.0
    checkpoint: str | None = None
    device: str = "cpu"


@register("policy", "dqn_beer_game")
class DQNBeerGamePolicy(OrderingPolicy):
    """DQN ordering policy (Oroojlooyjadid et al., 2022)."""

    def __init__(
        self,
        lead_time: int = 2,
        service_level: float = 0.95,
        *,
        checkpoint: str | None = None,
        action_offsets: list[int] | None = None,
        state_window: int = 5,
        hidden_sizes: tuple = (180, 130, 61, 33),
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.lead_time = int(lead_time)
        self.service_level = float(service_level)
        self.config = DQNBeerGameConfig(
            action_offsets=action_offsets or [-2, -1, 0, 1, 2],
            state_window=state_window,
            hidden_sizes=hidden_sizes,
            epsilon=epsilon,
            checkpoint=checkpoint,
            device=device,
        )

        w = self.config.state_window
        self._order_hist: deque[float] = deque([0.0] * w, maxlen=w)
        self._demand_hist: deque[float] = deque([0.0] * w, maxlen=w)
        self._shipment_hist: deque[float] = deque([0.0] * w, maxlen=w)
        self._inv_hist: deque[float] = deque([0.0] * w, maxlen=w)

        self._torch = None
        self._device = None
        self._qnet = None
        self._rng = np.random.default_rng(0)
        if checkpoint is not None:
            self._load_model()

    # ------------------------------------------------------------------ API
    def compute_order(
        self,
        inventory_position: float,
        forecast_mean: float,
        forecast_std: float,
    ) -> float:
        # Fallback when no trained weights are available.
        if self._qnet is None:
            # Myopic order-up-to with lead-time aware buffer.
            S = (self.lead_time + 1) * forecast_mean + 1.645 * forecast_std * np.sqrt(
                self.lead_time + 1
            )
            return float(max(0.0, S - inventory_position))

        torch = self._torch
        state = self._build_state(inventory_position, forecast_mean)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            q = self._qnet(state_t.unsqueeze(0)).squeeze(0).cpu().numpy()

        if self._rng.random() < self.config.epsilon:
            idx = int(self._rng.integers(0, len(self.config.action_offsets)))
        else:
            idx = int(np.argmax(q))

        offset = self.config.action_offsets[idx]
        order = max(0.0, forecast_mean + float(offset))
        self._order_hist.append(order)
        return float(order)

    def update_observation(
        self,
        *,
        demand_received: float,
        shipment_received: float,
        on_hand: float,
    ) -> None:
        self._demand_hist.append(float(demand_received))
        self._shipment_hist.append(float(shipment_received))
        self._inv_hist.append(float(on_hand))

    def reset(self) -> None:
        w = self.config.state_window
        self._order_hist = deque([0.0] * w, maxlen=w)
        self._demand_hist = deque([0.0] * w, maxlen=w)
        self._shipment_hist = deque([0.0] * w, maxlen=w)
        self._inv_hist = deque([0.0] * w, maxlen=w)

    # -------------------------------------------------------------- internals
    def _build_state(self, ip: float, fm: float) -> np.ndarray:
        return np.concatenate([
            np.array([ip, fm, float(self.lead_time)], dtype=np.float32),
            np.asarray(self._order_hist, dtype=np.float32),
            np.asarray(self._demand_hist, dtype=np.float32),
            np.asarray(self._shipment_hist, dtype=np.float32),
            np.asarray(self._inv_hist, dtype=np.float32),
        ])

    def _load_model(self) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DQNBeerGamePolicy(checkpoint=...) requires torch. "
                "Install with `pip install deepbullwhip[learning]`."
            ) from exc

        self._torch = torch
        self._device = torch.device(self.config.device)

        in_dim = 3 + 4 * self.config.state_window
        layers = []
        prev = in_dim
        for h in self.config.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, len(self.config.action_offsets))]
        self._qnet = nn.Sequential(*layers).to(self._device)

        state_dict = torch.load(self.config.checkpoint, map_location=self._device)
        self._qnet.load_state_dict(state_dict)
        self._qnet.eval()
