"""Vectorized supply chain simulation engine.

Operates on (N, K, T) matrices for Monte Carlo batching. The time loop
is sequential (AR/inventory state dependency), but all N paths and
the per-echelon computations within each time step are fully vectorized.

Typical speedup: 20-50x over N sequential SerialSupplyChain.simulate()
calls for N >= 100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from deepbullwhip._types import EchelonResult, SimulationResult
from deepbullwhip.chain.config import EchelonConfig, default_semiconductor_config


@dataclass
class BatchSimulationResult:
    """Results from a vectorized batch simulation.

    All arrays have a leading N (number of paths) dimension.
    """

    orders: np.ndarray          # (N, K, T)
    inventory: np.ndarray       # (N, K, T)
    costs: np.ndarray           # (N, K, T)
    bullwhip_ratios: np.ndarray # (N, K)
    fill_rates: np.ndarray      # (N, K)
    total_costs: np.ndarray     # (N, K)
    cumulative_bullwhip: np.ndarray  # (N,)

    @property
    def n_paths(self) -> int:
        return self.orders.shape[0]

    @property
    def n_echelons(self) -> int:
        return self.orders.shape[1]

    @property
    def n_periods(self) -> int:
        return self.orders.shape[2]

    def mean_metrics(self) -> dict[str, float]:
        """Average metrics across all N paths."""
        d: dict[str, float] = {}
        K = self.n_echelons
        for k in range(K):
            d[f"BW_{k + 1}"] = float(self.bullwhip_ratios[:, k].mean())
            d[f"cost_{k + 1}"] = float(self.total_costs[:, k].mean())
            d[f"fill_rate_{k + 1}"] = float(self.fill_rates[:, k].mean())
        d["BW_cumulative"] = float(self.cumulative_bullwhip.mean())
        d["total_cost"] = float(self.total_costs.sum(axis=1).mean())
        return d

    def to_simulation_result(self, path_index: int = 0) -> SimulationResult:
        """Extract a single path as a standard SimulationResult."""
        echelon_results = []
        for k in range(self.n_echelons):
            echelon_results.append(
                EchelonResult(
                    name=f"E{k + 1}",
                    orders=self.orders[path_index, k],
                    inventory_levels=self.inventory[path_index, k],
                    costs=self.costs[path_index, k],
                    bullwhip_ratio=float(self.bullwhip_ratios[path_index, k]),
                    fill_rate=float(self.fill_rates[path_index, k]),
                    total_cost=float(self.total_costs[path_index, k]),
                )
            )
        return SimulationResult(
            echelon_results=echelon_results,
            cumulative_bullwhip=float(self.cumulative_bullwhip[path_index]),
            total_cost=float(self.total_costs[path_index].sum()),
        )


class VectorizedSupplyChain:
    """Matrix-based supply chain simulation for Monte Carlo batching.

    Instead of Python lists and per-element operations, this engine
    pre-allocates (N, K, T) arrays and uses NumPy broadcasting for all
    N paths simultaneously. The pipeline is implemented as a circular
    buffer with O(1) index arithmetic instead of O(L) list.pop(0).

    Parameters
    ----------
    configs : list[EchelonConfig] or None
        Echelon configurations. If None, uses default semiconductor config.
    """

    def __init__(self, configs: list[EchelonConfig] | None = None) -> None:
        if configs is None:
            configs = default_semiconductor_config()
        self.configs = configs
        self.K = len(configs)

        # Pre-compute per-echelon parameters as arrays for vectorized access
        self.lead_times = np.array([c.lead_time for c in configs])
        self.L_max = int(self.lead_times.max())

        self.h = np.array([c.holding_cost + c.depreciation_rate for c in configs])
        self.b = np.array([c.backorder_cost for c in configs])
        self.z_alpha = np.array([stats.norm.ppf(c.service_level) for c in configs])
        self.initial_inv = np.array([c.initial_inventory for c in configs])

    def simulate(
        self,
        demand: np.ndarray,
        forecasts_mean: np.ndarray,
        forecasts_std: np.ndarray,
    ) -> BatchSimulationResult:
        """Run vectorized simulation.

        Parameters
        ----------
        demand : array, shape (N, T) or (T,)
            If 1-D, broadcast to all N paths (N=1).
        forecasts_mean : array, shape (N, T) or (T,)
        forecasts_std : array, shape (N, T) or (T,)

        Returns
        -------
        BatchSimulationResult
        """
        # Normalize to 2-D
        if demand.ndim == 1:
            demand = demand[np.newaxis, :]
            forecasts_mean = forecasts_mean[np.newaxis, :]
            forecasts_std = forecasts_std[np.newaxis, :]

        N, T = demand.shape
        K = self.K

        # ── Pre-allocate all matrices ────────────────────────────────
        orders = np.zeros((N, K, T))
        inventory = np.zeros((N, K, T))
        costs = np.zeros((N, K, T))

        # Pipeline: circular buffer (N, K, L_max)
        pipeline = np.zeros((N, K, self.L_max))
        # On-hand inventory: (N, K)
        inv = np.tile(self.initial_inv, (N, 1))  # (N, K)

        # Pipeline read pointer per echelon (wraps with %)
        ptr = np.zeros(K, dtype=int)

        # Lead time masks for circular buffer indexing
        # lead_time_mask[k, l] = True if l < lead_times[k]
        lead_time_mask = np.arange(self.L_max)[None, :] < self.lead_times[:, None]  # (K, L_max)

        # ── Time loop (sequential — state dependency) ────────────────
        for t in range(T):
            # --- Receive from pipeline: oldest order arrives ---
            # For each echelon k, read pipeline[n, k, ptr[k]]
            incoming = pipeline[:, np.arange(K), ptr]  # (N, K)
            inv += incoming

            # --- Compute inventory position: on-hand + pipeline sum ---
            # Mask out unused pipeline slots (beyond lead_time)
            pipeline_masked = pipeline * lead_time_mask[None, :, :]  # (N, K, L_max)
            pipeline_sum = pipeline_masked.sum(axis=2)  # (N, K)
            ip = inv + pipeline_sum  # (N, K)

            # --- Compute forecasts for each echelon ---
            # Echelon 0: use provided forecasts
            # Echelons 1..K-1: rolling mean/std of downstream orders (last 8)
            fm = np.zeros((N, K))
            fs = np.zeros((N, K))
            fm[:, 0] = forecasts_mean[:, t]
            fs[:, 0] = forecasts_std[:, t]

            if t > 0:
                window = min(8, t)
                for k in range(1, K):
                    recent = orders[:, k - 1, t - window : t]  # (N, window)
                    fm[:, k] = recent.mean(axis=1)
                    fs[:, k] = recent.std(axis=1) if window > 1 else forecasts_std[:, t]
            else:
                fm[:, 1:] = forecasts_mean[:, t : t + 1]
                fs[:, 1:] = forecasts_std[:, t : t + 1]

            # --- Order-Up-To policy (vectorized across N and K) ---
            L_plus_1 = (self.lead_times + 1).astype(float)  # (K,)
            S = L_plus_1[None, :] * fm + (
                self.z_alpha[None, :] * fs * np.sqrt(L_plus_1)[None, :]
            )  # (N, K)
            order_qty = np.maximum(0.0, S - ip)  # (N, K)

            # --- Write order into pipeline at current pointer ---
            pipeline[:, np.arange(K), ptr] = order_qty
            orders[:, :, t] = order_qty

            # --- Satisfy demand ---
            # Echelon 0 faces customer demand; echelon k faces echelon k-1 orders
            echelon_demand = np.zeros((N, K))
            echelon_demand[:, 0] = demand[:, t]
            if K > 1:
                echelon_demand[:, 1:] = order_qty[:, :-1]

            inv -= echelon_demand

            # --- Compute costs (newsvendor: h * inv+ + b * inv-) ---
            costs_t = np.where(inv >= 0,
                               self.h[None, :] * inv,
                               self.b[None, :] * np.abs(inv))
            costs[:, :, t] = costs_t
            inventory[:, :, t] = inv

            # --- Advance pipeline pointer (circular buffer) ---
            ptr = (ptr + 1) % self.L_max

        # ── Compute metrics (fully vectorized) ───────────────────────
        var_demand = np.var(demand, axis=1)  # (N,)

        # Variance of orders: (N, K)
        var_orders = np.var(orders, axis=2)

        # Bullwhip ratios
        bullwhip_ratios = np.ones((N, K))
        safe_var_demand = np.where(var_demand > 0, var_demand, 1.0)
        bullwhip_ratios[:, 0] = var_orders[:, 0] / safe_var_demand

        for k in range(1, K):
            safe_var_prev = np.where(var_orders[:, k - 1] > 0, var_orders[:, k - 1], 1.0)
            bullwhip_ratios[:, k] = var_orders[:, k] / safe_var_prev

        # Fill rates: (N, K)
        fill_rates = np.mean(inventory >= 0, axis=2)

        # Total costs: (N, K)
        total_costs = costs.sum(axis=2)

        # Cumulative bullwhip: (N,)
        var_last = var_orders[:, -1]
        cumulative_bullwhip = var_last / safe_var_demand

        return BatchSimulationResult(
            orders=orders,
            inventory=inventory,
            costs=costs,
            bullwhip_ratios=bullwhip_ratios,
            fill_rates=fill_rates,
            total_costs=total_costs,
            cumulative_bullwhip=cumulative_bullwhip,
        )
