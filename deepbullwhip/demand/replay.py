"""Replay historical demand data as a demand generator."""

from __future__ import annotations

import numpy as np

from deepbullwhip._types import TimeSeries
from deepbullwhip.demand.base import DemandGenerator
from deepbullwhip.registry import register


@register("demand", "replay")
class ReplayDemandGenerator(DemandGenerator):
    """Replay a historical demand series.

    Cycles the data if T exceeds the data length. For batch generation,
    adds small Gaussian noise (5% of data std) per path.

    Parameters
    ----------
    data : array-like
        Historical demand values, shape (T_data,).
    """

    def __init__(self, data: np.ndarray | list[float]) -> None:
        self.data = np.asarray(data, dtype=np.float64)

    def generate(self, T: int, seed: int | None = None) -> TimeSeries:
        return np.tile(self.data, (T // len(self.data)) + 1)[:T]

    def generate_batch(
        self, T: int = 156, n_paths: int = 100, seed: int | None = None
    ) -> np.ndarray:
        base = self.generate(T)
        rng = np.random.default_rng(seed)
        noise_scale = self.data.std() * 0.05 if self.data.std() > 0 else 0.0
        noise = rng.normal(0, noise_scale, (n_paths, T))
        return np.maximum(0, base[None, :] + noise)
