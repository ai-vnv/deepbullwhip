"""Predefined supply chain configurations for benchmarking."""

from deepbullwhip.chain.config import EchelonConfig, default_semiconductor_config

PREDEFINED_CONFIGS: dict[str, list[EchelonConfig]] = {
    "semiconductor_4tier": default_semiconductor_config(),
    "beer_game": [
        EchelonConfig(
            "Retailer", lead_time=2,
            holding_cost=0.50, backorder_cost=1.00,
        ),
        EchelonConfig(
            "Wholesaler", lead_time=2,
            holding_cost=0.50, backorder_cost=1.00,
        ),
        EchelonConfig(
            "Distributor", lead_time=2,
            holding_cost=0.50, backorder_cost=1.00,
        ),
        EchelonConfig(
            "Factory", lead_time=2,
            holding_cost=0.50, backorder_cost=1.00,
        ),
    ],
    "consumer_2tier": [
        EchelonConfig(
            "Retailer", lead_time=1,
            holding_cost=0.20, backorder_cost=0.80,
        ),
        EchelonConfig(
            "Manufacturer", lead_time=4,
            holding_cost=0.10, backorder_cost=0.40,
        ),
    ],
}
