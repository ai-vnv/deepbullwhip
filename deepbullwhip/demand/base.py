from abc import ABC, abstractmethod

from deepbullwhip._types import TimeSeries


class DemandGenerator(ABC):
    """Abstract base class for demand generators."""

    @abstractmethod
    def generate(self, T: int, seed: int | None = None) -> TimeSeries:
        """Generate a demand time series of length T.

        Parameters
        ----------
        T : int
            Number of periods.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        TimeSeries
            1-D array of non-negative demand values, shape (T,).
        """
        ...
