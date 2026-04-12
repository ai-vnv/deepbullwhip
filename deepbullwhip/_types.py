from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

TimeSeries = NDArray[np.float64]


@dataclass
class EchelonResult:
    """Results for a single echelon after simulation."""

    name: str
    orders: TimeSeries
    inventory_levels: TimeSeries
    costs: TimeSeries
    bullwhip_ratio: float
    fill_rate: float
    total_cost: float


@dataclass
class SimulationResult:
    """Aggregate results from a full supply chain simulation."""

    echelon_results: list[EchelonResult]
    cumulative_bullwhip: float
    total_cost: float

    def to_dict(self) -> dict[str, float]:
        """Flat dictionary matching the notebook's get_metrics() output."""
        d: dict[str, float] = {}
        for k, er in enumerate(self.echelon_results):
            d[f"BW_{k + 1}"] = er.bullwhip_ratio
            d[f"cost_{k + 1}"] = er.total_cost
            d[f"fill_rate_{k + 1}"] = er.fill_rate
        d["BW_cumulative"] = self.cumulative_bullwhip
        d["total_cost"] = self.total_cost
        return d


@dataclass
class NetworkSimulationResult:
    """Results from a network (DAG) supply chain simulation.

    Unlike :class:`SimulationResult` which uses positional indexing,
    this stores per-node results keyed by node name, supporting
    arbitrary DAG topologies.

    Parameters
    ----------
    node_results : dict[str, EchelonResult]
        Per-node simulation results, keyed by node name.
    edge_flows : dict[tuple[str, str], TimeSeries]
        Material flow time series for each edge ``(upstream, downstream)``.
    cumulative_bullwhip : float
        Maximum bullwhip ratio across all demand-facing nodes.
    total_cost : float
        Sum of costs across all nodes.
    """

    node_results: dict[str, EchelonResult]
    edge_flows: dict[tuple[str, str], TimeSeries]
    cumulative_bullwhip: float
    total_cost: float

    def to_simulation_result(self) -> SimulationResult:
        """Convert to a :class:`SimulationResult` for backward compatibility.

        Node results are ordered by name.

        Returns
        -------
        SimulationResult
        """
        echelon_results = list(self.node_results.values())
        return SimulationResult(
            echelon_results=echelon_results,
            cumulative_bullwhip=self.cumulative_bullwhip,
            total_cost=self.total_cost,
        )

    def to_dict(self) -> dict[str, float]:
        """Flat dictionary of metrics keyed by node name.

        Returns
        -------
        dict[str, float]
            Keys like ``"BW_Factory"``, ``"cost_Retailer"``, etc.
        """
        d: dict[str, float] = {}
        for name, er in self.node_results.items():
            d[f"BW_{name}"] = er.bullwhip_ratio
            d[f"cost_{name}"] = er.total_cost
            d[f"fill_rate_{name}"] = er.fill_rate
        d["BW_cumulative"] = self.cumulative_bullwhip
        d["total_cost"] = self.total_cost
        return d
