from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EchelonConfig:
    """Configuration for a single supply chain echelon."""

    name: str
    lead_time: int
    holding_cost: float
    backorder_cost: float
    depreciation_rate: float = 0.0
    service_level: float = 0.95
    initial_inventory: float = 50.0


def default_semiconductor_config() -> list[EchelonConfig]:
    """Return the 4-echelon semiconductor config from the paper.

    Weekly depreciation derived from 15% quarterly value loss:
        weekly_dep = 1 - (1 - 0.15) ** (1/13)
    """
    weekly_dep = 1 - (1 - 0.15) ** (1 / 13)
    return [
        EchelonConfig(
            "Distributor", lead_time=2, holding_cost=0.15,
            backorder_cost=0.60, depreciation_rate=weekly_dep,
        ),
        EchelonConfig(
            "OSAT", lead_time=4, holding_cost=0.12,
            backorder_cost=0.50, depreciation_rate=weekly_dep * 0.8,
        ),
        EchelonConfig(
            "Foundry", lead_time=12, holding_cost=0.08,
            backorder_cost=0.40, depreciation_rate=weekly_dep * 0.5,
        ),
        EchelonConfig(
            "Supplier", lead_time=8, holding_cost=0.05,
            backorder_cost=0.30, depreciation_rate=weekly_dep * 0.3,
        ),
    ]
