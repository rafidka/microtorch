from typing import Any

from microtorch.nn import Module, Parameter
from microtorch.nn.modules.utils import NamedParameterIterator, ParameterIterator

# pyright: reportPrivateUsage=false


class DummyParameter(Parameter):
    def __init__(self, name: str, value: Any):
        super().__init__(value)
        self.name = name
        self.value = value


def make_simple_model():
    model = Module[int]()
    model._parameters["w1"] = DummyParameter("w1", 1)
    model._parameters["b1"] = DummyParameter("b1", 2)

    child = Module[int]()
    child._parameters["w2"] = DummyParameter("w2", 3)
    model._modules["child"] = child

    return model


def test_parameter_iterator_basic():
    model = make_simple_model()
    params = list(ParameterIterator(model))
    assert len(params) == 3
    assert all(isinstance(p, Parameter) for p in params)


def test_parameter_iterator_no_recurse():
    model = make_simple_model()
    params = list(ParameterIterator(model, recurse=False))
    assert len(params) == 2  # Only top-level parameters


def test_parameter_iterator_remove_duplicates():
    model = Module[int]()
    shared = DummyParameter("shared", 42)
    model._parameters["a"] = shared
    child = Module[int]()
    child._parameters["b"] = shared  # same param
    model._modules["child"] = child

    params = list(ParameterIterator(model, recurse=True, remove_duplicate=True))
    assert len(params) == 1

    params_all = list(ParameterIterator(model, recurse=True, remove_duplicate=False))
    assert len(params_all) == 2


def test_named_parameter_iterator_basic():
    model = make_simple_model()
    named_params = dict(NamedParameterIterator(model))
    assert set(named_params.keys()) == {"w1", "b1", "child.w2"}


def test_named_parameter_iterator_no_recurse():
    model = make_simple_model()
    named_params = dict(NamedParameterIterator(model, recurse=False))
    assert set(named_params.keys()) == {"w1", "b1"}


def test_named_parameter_iterator_remove_duplicates():
    model = Module[int]()
    shared = DummyParameter("shared", 42)
    model._parameters["a"] = shared
    child = Module[int]()
    child._parameters["b"] = shared
    model._modules["child"] = child

    named = dict(NamedParameterIterator(model, recurse=True, remove_duplicate=True))
    assert len(named) == 1
    assert "a" in named or "child.b" in named

    named_all = dict(
        NamedParameterIterator(model, recurse=True, remove_duplicate=False)
    )
    assert len(named_all) == 2
    assert "a" in named_all and "child.b" in named_all


def test_iterators_are_reusable():
    model = make_simple_model()
    it = ParameterIterator(model)
    list1 = list(it)
    list2 = list(it)
    assert list1 == list2

    it_named = NamedParameterIterator(model)
    list1 = list(it_named)
    list2 = list(it_named)
    assert dict(list1) == dict(list2)
