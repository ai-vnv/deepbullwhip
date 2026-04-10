"""Cost metrics."""

from deepbullwhip._types import SimulationResult, TimeSeries
from deepbullwhip.registry import register


@register("metric", "TC")
class TotalCost:
    """Total Cost: sum of per-period costs across all echelons."""

    @staticmethod
    def compute(
        result: SimulationResult,
        demand: TimeSeries,
        echelon: int = 0,
    ) -> float:
        er = result.echelon_results[echelon]
        return float(er.total_cost)
