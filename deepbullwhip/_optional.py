"""Lazy import utility for optional dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType


def import_optional(package_name: str, extra: str) -> ModuleType:
    """Import an optional dependency, raising a clear error if missing.

    Parameters
    ----------
    package_name : str
        The package to import (e.g. ``"networkx"``).
    extra : str
        The pip extra that provides the package (e.g. ``"network"``).

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the package is not installed, with install instructions.
    """
    try:
        return importlib.import_module(package_name)
    except ImportError:
        raise ImportError(
            f"{package_name} is required for this feature. "
            f"Install it with: pip install deepbullwhip[{extra}]"
        ) from None
