"""Learning-based ordering policies (optional heavier dependencies).

Import via ``import deepbullwhip.ext`` to register all extension
components with :mod:`deepbullwhip.registry`.
"""

from . import dcl  # noqa: F401
from . import dqn_beer_game  # noqa: F401
from . import e2e_newsvendor  # noqa: F401
from . import recurrent_ppo  # noqa: F401

__all__: list[str] = []
