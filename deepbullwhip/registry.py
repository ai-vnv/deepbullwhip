"""Decorator-based model registry for DeepBullwhip components."""

from __future__ import annotations

from typing import Any

_REGISTRY: dict[str, dict[str, type]] = {
    "demand": {},
    "policy": {},
    "cost": {},
    "forecaster": {},
    "metric": {},
}


def register(category: str, name: str):
    """Class decorator to register a component.

    Usage::

        @register("policy", "proportional_out")
        class ProportionalOUTPolicy(OrderingPolicy):
            ...

    Parameters
    ----------
    category : str
        One of "demand", "policy", "cost", "forecaster", "metric".
    name : str
        Unique name within the category.
    """

    def decorator(cls):
        if category not in _REGISTRY:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Valid: {list(_REGISTRY.keys())}"
            )
        _REGISTRY[category][name] = cls
        cls._registry_key = f"{category}/{name}"
        cls._registry_name = name
        return cls

    return decorator


def get(category: str, name: str, **kwargs: Any) -> Any:
    """Instantiate a registered class by category and name.

    Parameters
    ----------
    category : str
        Component category.
    name : str
        Registered name.
    **kwargs
        Passed to the class constructor.

    Returns
    -------
    Instance of the registered class.

    Raises
    ------
    KeyError
        If the name is not found in the category.
    """
    try:
        cls = _REGISTRY[category][name]
    except KeyError:
        available = list(_REGISTRY.get(category, {}).keys())
        raise KeyError(
            f"'{name}' not found in category '{category}'. "
            f"Available: {available}"
        ) from None
    return cls(**kwargs)


def get_class(category: str, name: str) -> type:
    """Return the registered class (without instantiating).

    Parameters
    ----------
    category : str
        Component category.
    name : str
        Registered name.

    Returns
    -------
    type
        The registered class.
    """
    try:
        return _REGISTRY[category][name]
    except KeyError:
        available = list(_REGISTRY.get(category, {}).keys())
        raise KeyError(
            f"'{name}' not found in category '{category}'. "
            f"Available: {available}"
        ) from None


def list_registered(
    category: str | None = None,
) -> dict[str, list[str]] | list[str]:
    """List registered component names.

    Parameters
    ----------
    category : str or None
        If given, return names in that category.
        If None, return dict of all categories.

    Returns
    -------
    list[str] or dict[str, list[str]]
    """
    if category is not None:
        return list(_REGISTRY.get(category, {}).keys())
    return {cat: list(entries.keys()) for cat, entries in _REGISTRY.items()}
