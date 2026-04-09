from abc import ABC, abstractmethod


class CostFunction(ABC):
    """Abstract per-period cost function."""

    @abstractmethod
    def compute(self, inventory: float) -> float:
        """Compute cost for a single period given ending inventory.

        Parameters
        ----------
        inventory : float
            On-hand inventory (negative means backorders).

        Returns
        -------
        float
            Non-negative cost for this period.
        """
        ...
