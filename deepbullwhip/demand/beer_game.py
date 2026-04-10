"""Classic MIT Beer Game step demand generator."""

import numpy as np

from deepbullwhip._types import TimeSeries
from deepbullwhip.demand.base import DemandGenerator
from deepbullwhip.registry import register


@register("demand", "beer_game")
class BeerGameDemandGenerator(DemandGenerator):
    """Classic MIT Beer Game demand: constant base, then step up.

    Parameters
    ----------
    base_demand : float
        Demand before the step (default 4.0).
    step_demand : float
        Demand after the step (default 8.0).
    step_time : int
        Period at which the step occurs (default 5).
    """

    def __init__(
        self,
        base_demand: float = 4.0,
        step_demand: float = 8.0,
        step_time: int = 5,
    ) -> None:
        self.base_demand = base_demand
        self.step_demand = step_demand
        self.step_time = step_time

    def generate(self, T: int, seed: int | None = None) -> TimeSeries:
        d = np.full(T, self.base_demand)
        d[self.step_time :] = self.step_demand
        return d

    def generate_batch(
        self, T: int = 52, n_paths: int = 100, seed: int | None = None
    ) -> np.ndarray:
        """All paths are identical (deterministic demand)."""
        return np.tile(self.generate(T), (n_paths, 1))
