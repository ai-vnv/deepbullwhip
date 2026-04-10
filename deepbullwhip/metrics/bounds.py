"""Theoretical bullwhip ratio bounds."""

from deepbullwhip.registry import register


@register("metric", "ChenLowerBound")
class ChenLowerBound:
    """Chen et al. (2000) Theorem 1 lower bound on BWR.

    BWR >= 1 + 2*L*lambda*phi / (1 + phi^2) + L^2 * lambda^2

    Parameters
    ----------
    lead_time : int
    sensitivity : float (lambda_f)
    phi : float (AR(1) coefficient)
    """

    def __init__(
        self,
        lead_time: int = 2,
        sensitivity: float = 1.0,
        phi: float = 0.72,
    ) -> None:
        self.lead_time = lead_time
        self.sensitivity = sensitivity
        self.phi = phi

    def compute_bound(self) -> float:
        """Compute the lower bound value."""
        L = self.lead_time
        lam = self.sensitivity
        return (
            1
            + 2 * L * lam * self.phi / (1 + self.phi**2)
            + L**2 * lam**2
        )
