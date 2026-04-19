"""Neural and gradient-boosting forecasters (optional heavier dependencies).

Import via ``import deepbullwhip.ext`` to register all extension
components with :mod:`deepbullwhip.registry`.
"""

from . import lightgbm_quantile  # noqa: F401
from . import lstm_multistep  # noqa: F401
from . import nbeats  # noqa: F401
from . import tft  # noqa: F401

__all__: list[str] = []
