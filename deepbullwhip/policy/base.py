from abc import ABC, abstractmethod


class OrderingPolicy(ABC):
    """Abstract ordering policy for a single echelon."""

    @abstractmethod
    def compute_order(
        self,
        inventory_position: float,
        forecast_mean: float,
        forecast_std: float,
    ) -> float:
        """Compute the non-negative order quantity for this period.

        Parameters
        ----------
        inventory_position : float
            On-hand inventory + pipeline - backorders.
        forecast_mean : float
            Point forecast of next-period demand.
        forecast_std : float
            Standard deviation of forecast error.

        Returns
        -------
        float
            Non-negative order quantity.
        """
        ...
