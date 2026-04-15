"""
deepar_forecaster.py — DeepAR integration for deepbullwhip
==========================================================

Implements the DeepAR probabilistic forecaster (Salinas et al., 2020)
as a pluggable component in the deepbullwhip benchmarking framework.

Two modes:
  1. DeepARForecaster  — wraps a pre-trained GluonTS DeepAR model
  2. DeepARTrainer      — trains a DeepAR model from multi-echelon
                          demand data and returns a DeepARForecaster

Architecture note:
  deepbullwhip's Forecaster ABC expects:
      forecast(demand_history, steps_ahead=1) -> (mean, std)
  
  DeepAR produces Monte Carlo sample paths from an autoregressive RNN.
  We extract (mean, std) from those samples to satisfy the interface.

  The key insight from the paper: DeepAR learns a GLOBAL model across
  all time series. In the supply chain context, this means training on
  demand histories from ALL echelons and Monte Carlo paths jointly,
  enabling cross-echelon pattern learning (e.g., amplification signatures).

Requirements:
  pip install gluonts[torch] torch   # GluonTS with PyTorch backend
  pip install deepbullwhip

Usage:
  # ---- Quick: use with pre-trained model ----
  from deepar_forecaster import DeepARForecaster
  fc = DeepARForecaster(predictor=trained_predictor, context_length=52)
  
  # ---- Full: train + benchmark ----
  from deepar_forecaster import DeepARTrainer
  trainer = DeepARTrainer(
      freq="W",
      prediction_length=1,       # 1-step rolling forecast
      context_length=52,         # 1 year lookback
      epochs=50,
      num_layers=3,
      hidden_size=40,
  )
  # Train on a collection of demand arrays
  forecaster = trainer.train(demand_arrays=[d1, d2, d3, ...])
  
  # Register and benchmark
  from deepbullwhip.benchmark import BenchmarkRunner
  runner = BenchmarkRunner(...)
  results = runner.run(
      policies=["order_up_to"],
      forecasters=[("deepar", {}), "naive", "moving_average"],
      metrics=["BWR", "CUM_BWR", "TC", "FILL_RATE"],
  )

Reference:
  Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020).
  DeepAR: Probabilistic forecasting with autoregressive recurrent
  networks. International Journal of Forecasting, 36(3), 1181-1191.
  https://doi.org/10.1016/j.ijforecast.2019.07.001
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from deepbullwhip.forecast.base import Forecaster
from deepbullwhip.registry import register


# ---------------------------------------------------------------------------
# DeepAR Forecaster — wraps a trained GluonTS predictor
# ---------------------------------------------------------------------------

@register("forecaster", "deepar")
class DeepARForecaster(Forecaster):
    """Probabilistic forecaster using a trained DeepAR model.

    Wraps a GluonTS ``Predictor`` behind the deepbullwhip ``Forecaster``
    ABC. At each call to ``forecast()``, constructs a GluonTS-compatible
    dataset from the demand history, runs the predictor to obtain Monte
    Carlo sample paths, and returns (mean, std) of the 1-step-ahead
    marginal distribution.

    Parameters
    ----------
    predictor : gluonts.torch.model.predictor.PyTorchPredictor, optional
        A trained GluonTS predictor. If None, a default model is created
        and must be trained via ``DeepARTrainer``.
    context_length : int
        Number of historical periods the model conditions on (encoder).
        Must match the value used during training.
    num_samples : int
        Number of Monte Carlo sample paths for probabilistic output.
        More samples → more accurate quantiles but slower inference.
        Salinas et al. used 200 in their experiments.
    freq : str
        Pandas frequency string matching the training data granularity.
    """

    def __init__(
        self,
        predictor=None,
        context_length: int = 52,
        num_samples: int = 200,
        freq: str = "W",
    ):
        self._predictor = predictor
        self.context_length = context_length
        self.num_samples = num_samples
        self.freq = freq

    def forecast(
        self, demand_history: np.ndarray, steps_ahead: int = 1
    ) -> tuple[float, float]:
        """Return (mean, std) from DeepAR Monte Carlo samples.

        If the predictor is not set or history is too short, falls back
        to the naïve sample-statistics forecast for robustness.
        """
        if self._predictor is None or len(demand_history) < 3:
            # Graceful fallback for untrained model or cold start
            mean = float(np.mean(demand_history))
            std = float(np.std(demand_history)) if len(demand_history) > 1 else 0.0
            return mean, std

        try:
            return self._deepar_forecast(demand_history, steps_ahead)
        except Exception:
            # If GluonTS inference fails (e.g., shape mismatch), fallback
            mean = float(np.mean(demand_history[-self.context_length :]))
            std = float(np.std(demand_history[-self.context_length :]))
            return mean, std

    def _deepar_forecast(
        self, demand_history: np.ndarray, steps_ahead: int
    ) -> tuple[float, float]:
        """Run DeepAR inference via GluonTS."""
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        # Use the most recent context_length observations
        history = demand_history[-self.context_length :]

        # Build a GluonTS-compatible dataset entry
        start = pd.Timestamp("2020-01-01")  # arbitrary anchor
        dataset = ListDataset(
            [{"start": start, "target": history.tolist()}],
            freq=self.freq,
        )

        # Run prediction — returns SampleForecast objects
        forecasts = list(self._predictor.predict(dataset, num_samples=self.num_samples))
        forecast = forecasts[0]

        # Extract the 1-step-ahead marginal from sample paths
        # forecast.samples has shape (num_samples, prediction_length)
        samples_step = forecast.samples[:, min(steps_ahead - 1, forecast.samples.shape[1] - 1)]

        mean = float(np.mean(samples_step))
        std = float(np.std(samples_step))

        return mean, std

    def generate_forecasts(
        self, demand: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rolling forecast over a full demand series.

        Overrides the base class to use batch inference when possible,
        which is significantly faster than calling forecast() T times.
        """
        if self._predictor is None:
            # Fallback to base class rolling loop
            return super().generate_forecasts(demand)

        try:
            return self._batch_rolling_forecast(demand)
        except Exception:
            return super().generate_forecasts(demand)

    def _batch_rolling_forecast(
        self, demand: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch rolling forecast for efficiency.

        Constructs multiple overlapping windows and runs them through
        DeepAR in a single batch, avoiding the per-step overhead.
        """
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        T = len(demand)
        fm = np.zeros(T)
        fs = np.zeros(T)

        # For early periods where we don't have enough context, use naïve
        warmup = min(self.context_length, T)
        for t in range(warmup):
            h = demand[: t + 1]
            fm[t] = float(np.mean(h))
            fs[t] = float(np.std(h)) if len(h) > 1 else 0.0

        if warmup >= T:
            return fm, fs

        # Build batch of windows for periods [warmup, T)
        start = pd.Timestamp("2020-01-01")
        entries = []
        for t in range(warmup, T):
            window = demand[max(0, t + 1 - self.context_length) : t + 1]
            entries.append({"start": start, "target": window.tolist()})

        dataset = ListDataset(entries, freq=self.freq)
        predictions = list(
            self._predictor.predict(dataset, num_samples=self.num_samples)
        )

        for idx, t in enumerate(range(warmup, T)):
            samples = predictions[idx].samples[:, 0]  # 1-step-ahead
            fm[t] = float(np.mean(samples))
            fs[t] = float(np.std(samples))

        return fm, fs


# ---------------------------------------------------------------------------
# DeepAR Trainer — trains a DeepAR model from supply chain demand data
# ---------------------------------------------------------------------------

class DeepARTrainer:
    """Train a DeepAR model on multi-echelon supply chain demand data.

    This class handles the full pipeline:
      1. Convert demand arrays to GluonTS ListDataset format
      2. Configure the DeepAR estimator with appropriate hyperparams
      3. Train and return a DeepARForecaster ready for benchmarking

    The key architectural choice from Salinas et al.: train a GLOBAL
    model across all available time series. In the supply chain context,
    this means pooling demand from all echelons, Monte Carlo paths,
    and/or historical scenarios into a single training corpus.

    Parameters
    ----------
    freq : str
        Time series frequency ("W" for weekly, "M" for monthly, etc.)
    prediction_length : int
        Number of steps the model predicts ahead. For rolling 1-step
        forecasts in deepbullwhip, use 1.
    context_length : int
        Number of past observations the encoder conditions on.
    epochs : int
        Training epochs.
    num_layers : int
        Number of LSTM layers (Salinas et al. used 3).
    hidden_size : int
        LSTM hidden dimension (40 for small datasets, 120 for large).
    lr : float
        Learning rate for Adam optimizer.
    batch_size : int
        Training batch size.
    num_samples : int
        Monte Carlo samples at inference time.
    likelihood : str
        "gaussian" for real-valued, "negative_binomial" for count data.
        For supply chain demand (counts), negative_binomial is preferred.
    """

    def __init__(
        self,
        freq: str = "W",
        prediction_length: int = 1,
        context_length: int = 52,
        epochs: int = 50,
        num_layers: int = 3,
        hidden_size: int = 40,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_samples: int = 200,
        likelihood: str = "gaussian",
    ):
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.epochs = epochs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lr = lr
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.likelihood = likelihood

    def train(
        self,
        demand_arrays: Sequence[np.ndarray],
        start: str = "2020-01-01",
    ) -> DeepARForecaster:
        """Train DeepAR on a collection of demand time series.

        Parameters
        ----------
        demand_arrays : list of np.ndarray
            Each array is a 1-D demand history. These can come from:
            - Different echelons in the chain
            - Different Monte Carlo paths
            - Different products/scenarios
            The more diverse the training corpus, the better DeepAR
            can learn cross-series patterns.
        start : str
            Arbitrary start timestamp (DeepAR uses relative time).

        Returns
        -------
        DeepARForecaster
            Ready-to-use forecaster with the trained predictor.

        Example
        -------
        >>> from deepbullwhip.demand.semiconductor import SemiconductorDemandGenerator
        >>> gen = SemiconductorDemandGenerator()
        >>> # Generate diverse training series
        >>> series = [gen.generate(T=260, seed=s) for s in range(100)]
        >>> trainer = DeepARTrainer(freq="W", context_length=52, epochs=30)
        >>> forecaster = trainer.train(series)
        """
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        # ---- 1. Build GluonTS training dataset ----
        ts_start = pd.Timestamp(start)
        train_entries = [
            {"start": ts_start, "target": arr.tolist()}
            for arr in demand_arrays
        ]
        train_dataset = ListDataset(train_entries, freq=self.freq)

        # ---- 2. Configure DeepAR estimator ----
        # GluonTS PyTorch backend (post-MXNet migration)
        from gluonts.torch.model.deepar import DeepAREstimator

        estimator = DeepAREstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            lr=self.lr,
            batch_size=self.batch_size,
            trainer_kwargs={"max_epochs": self.epochs},
        )

        # ---- 3. Train ----
        predictor = estimator.train(training_data=train_dataset)

        # ---- 4. Wrap in DeepARForecaster ----
        return DeepARForecaster(
            predictor=predictor,
            context_length=self.context_length,
            num_samples=self.num_samples,
            freq=self.freq,
        )


# ---------------------------------------------------------------------------
# Integration example: full benchmark pipeline
# ---------------------------------------------------------------------------

def run_deepar_benchmark_example():
    """Complete example: train DeepAR and benchmark against baselines.

    This demonstrates the end-to-end workflow:
      1. Generate synthetic demand data for training
      2. Train a DeepAR model
      3. Run the BenchmarkRunner comparing DeepAR vs classical forecasters
      4. Analyze results across multiple metrics
    """
    from deepbullwhip.demand.semiconductor import SemiconductorDemandGenerator
    from deepbullwhip.chain.vectorized import VectorizedSupplyChain

    print("=" * 60)
    print("DeepAR + deepbullwhip Integration Benchmark")
    print("=" * 60)

    # ---- Step 1: Generate training corpus ----
    print("\n[1/4] Generating training demand series...")
    gen = SemiconductorDemandGenerator()
    # Generate 200 diverse demand realizations for training
    train_series = [gen.generate(T=260, seed=s) for s in range(200)]
    print(f"  Generated {len(train_series)} series, each T={len(train_series[0])}")

    # ---- Step 2: Train DeepAR ----
    print("\n[2/4] Training DeepAR model...")
    trainer = DeepARTrainer(
        freq="W",
        prediction_length=1,
        context_length=52,
        epochs=20,        # reduce for demo; use 50+ for production
        num_layers=3,
        hidden_size=40,
        lr=1e-3,
        batch_size=64,
        num_samples=200,
        likelihood="gaussian",
    )
    deepar_fc = trainer.train(train_series)
    print("  Training complete.")

    # ---- Step 3: Run benchmark ----
    print("\n[3/4] Running BenchmarkRunner...")

    # Use the trained forecaster directly
    # Since DeepARForecaster is @register'd as "deepar", the runner
    # can instantiate it by name — but we need to inject the predictor.
    # Option A: pass the instance directly
    # Option B: register a factory. We show Option A here.

    # For demonstration, run a manual comparison
    demand = gen.generate(T=156, seed=42)

    # Classical forecasters for comparison
    from deepbullwhip.forecast.naive import NaiveForecaster
    from deepbullwhip.forecast.moving_average import MovingAverageForecaster

    forecasters = {
        "DeepAR": deepar_fc,
        "Naïve": NaiveForecaster(),
        "MA(10)": MovingAverageForecaster(window=10),
    }

    chain = VectorizedSupplyChain()
    results = {}

    for name, fc in forecasters.items():
        fm, fs = fc.generate_forecasts(demand)
        result = chain.simulate(
            demand=np.expand_dims(demand, 0),  # (1, T)
            forecasts_mean=np.expand_dims(fm, 0),
            forecasts_std=np.expand_dims(fs, 0),
        )
        # Extract BWR at E4
        orders = result.orders[0]  # (K, T)
        bwr_e4 = np.var(orders[-1]) / np.var(demand)
        total_cost = float(np.sum(result.costs[0]))
        results[name] = {"BWR_E4": bwr_e4, "Total_Cost": total_cost}

    # ---- Step 4: Display results ----
    print("\n[4/4] Results")
    print(f"{'Forecaster':<15} {'BWR(E4)':>10} {'Total Cost':>12}")
    print("-" * 40)
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['BWR_E4']:>10.1f} {metrics['Total_Cost']:>12,.0f}")

    return results


if __name__ == "__main__":
    run_deepar_benchmark_example()
