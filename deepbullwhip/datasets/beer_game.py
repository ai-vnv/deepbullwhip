"""Classic Beer Game step demand dataset."""

import numpy as np

from deepbullwhip._types import TimeSeries


def load_beer_game(T: int = 52) -> TimeSeries:
    """Return the classic Beer Game step demand.

    Constant 4 for the first 5 periods, then 8 for the rest.

    Parameters
    ----------
    T : int
        Number of periods (default 52, one year of weekly data).

    Returns
    -------
    TimeSeries, shape (T,)
    """
    d = np.full(T, 4.0)
    d[5:] = 8.0
    return d
