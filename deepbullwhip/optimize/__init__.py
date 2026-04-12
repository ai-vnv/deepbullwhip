"""Supply chain optimization using Pyomo.

This module provides mathematical optimization models for supply chain
design and operation, integrating with DeepBullwhip's simulation engine
for simulation-optimization hybrid approaches.

Requires the ``pyomo`` optional dependency::

    pip install deepbullwhip[optimize]

Submodules
----------
inventory
    Multi-echelon inventory optimization (base-stock levels).
policy_tuning
    Simulation-based policy parameter tuning.
network_design
    Facility location and capacity allocation (experimental).

Functions
---------
build_inventory_model
    Build a Pyomo model for inventory optimization.
solve_model
    Solve a Pyomo model and extract results.
tune_service_levels
    Find optimal service levels via simulation-optimization.
tune_smoothing_factors
    Find optimal smoothing factors for SmoothingOUT policies.
build_network_design_model
    Build a facility location MIP (experimental).
"""

from deepbullwhip.optimize.inventory import build_inventory_model, solve_model
from deepbullwhip.optimize.network_design import build_network_design_model
from deepbullwhip.optimize.policy_tuning import (
    tune_service_levels,
    tune_smoothing_factors,
)

__all__ = [
    "build_inventory_model",
    "solve_model",
    "tune_service_levels",
    "tune_smoothing_factors",
    "build_network_design_model",
]
