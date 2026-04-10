"""Tests for the decorator-based model registry."""

import pytest

from deepbullwhip.registry import _REGISTRY, get, get_class, list_registered, register


class _DummyClass:
    """Dummy class for testing registration."""

    def __init__(self, x: int = 1):
        self.x = x


def test_register_and_get():
    register("demand", "_test_dummy")(_DummyClass)
    instance = get("demand", "_test_dummy", x=42)
    assert isinstance(instance, _DummyClass)
    assert instance.x == 42
    # Cleanup
    del _REGISTRY["demand"]["_test_dummy"]


def test_register_unknown_category():
    with pytest.raises(ValueError, match="Unknown category 'nonexistent'"):
        register("nonexistent", "foo")(_DummyClass)


def test_get_unknown_name():
    with pytest.raises(KeyError, match="'no_such_thing' not found"):
        get("demand", "no_such_thing")


def test_get_class_returns_type():
    cls = get_class("demand", "semiconductor_ar1")
    from deepbullwhip.demand.semiconductor import SemiconductorDemandGenerator

    assert cls is SemiconductorDemandGenerator


def test_get_class_unknown_name():
    with pytest.raises(KeyError, match="'no_such' not found"):
        get_class("policy", "no_such")


def test_list_registered_all():
    result = list_registered()
    assert isinstance(result, dict)
    assert "demand" in result
    assert "policy" in result
    assert "cost" in result
    assert "forecaster" in result
    assert "metric" in result


def test_list_registered_category():
    result = list_registered("demand")
    assert isinstance(result, list)
    assert "semiconductor_ar1" in result


def test_existing_classes_registered():
    assert "semiconductor_ar1" in list_registered("demand")
    assert "order_up_to" in list_registered("policy")
    assert "newsvendor" in list_registered("cost")


def test_registry_key_attribute():
    from deepbullwhip.demand.semiconductor import SemiconductorDemandGenerator

    assert SemiconductorDemandGenerator._registry_key == "demand/semiconductor_ar1"
    assert SemiconductorDemandGenerator._registry_name == "semiconductor_ar1"
