"""Built-in datasets for benchmarking supply chain simulations."""

from deepbullwhip.datasets.beer_game import load_beer_game
from deepbullwhip.datasets.loader import list_datasets, load_dataset
from deepbullwhip.datasets.m5 import load_m5
from deepbullwhip.datasets.synthetic import load_ar1, load_arma
from deepbullwhip.datasets.wsts import load_wsts

__all__ = [
    "load_beer_game",
    "load_ar1",
    "load_arma",
    "load_m5",
    "load_wsts",
    "load_dataset",
    "list_datasets",
]
