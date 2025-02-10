# Python imports
from typing import Any

# 3rd party imports
import pytest

# Local imports
from microtorch.nn import Module, Parameter

# pyright: reportPrivateUsage=false


class SimpleModule(Module[int]):
    """A simple test module that returns a constant value."""

    def __init__(self):
        super().__init__()
        self.param1 = Parameter([1.0])
        self.param2 = Parameter([2.0])

    def forward(self) -> int:
        return 123


class ModuleWithArgs(Module[tuple[int, str]]):
    """A test module that processes args and kwargs."""

    def forward(self, x: int, text: str = "default") -> tuple[int, str]:
        return (x * 2, text.upper())


class CompoundModule(Module[int]):
    def __init__(self):
        super().__init__()
        self.child1 = SimpleModule()
        self.child2 = SimpleModule()
        self.param1 = Parameter([10])
        self.param2 = Parameter([20])

    def forward(self) -> int:
        return self.child1() + self.child2()


def test_module_initialization():
    """Test that a new module is properly initialized."""
    module = Module[Any]()
    assert hasattr(module, "_modules")
    assert isinstance(module._modules, dict)
    assert len(module._modules) == 0


def test_add_module():
    """Test adding a submodule explicitly using add_module."""
    parent = Module[Any]()
    child = SimpleModule()

    parent.add_module("child", child)

    assert "child" in parent._modules
    assert parent._modules["child"] is child


def test_module_attribute_assignment():
    """Test that assigning a Module as an attribute adds it to _modules."""
    parent = Module[Any]()
    child = SimpleModule()

    parent.submodule = child

    assert "submodule" in parent._modules
    assert parent._modules["submodule"] is child
    assert parent.submodule is child  # type: ignore


def test_non_module_attribute_assignment():
    """Test that non-Module attributes are assigned normally."""
    module = Module[Any]()
    module.number = 123
    module.string = "test"

    assert module.number == 123  # type: ignore
    assert module.string == "test"  # type: ignore
    assert "number" not in module._modules
    assert "string" not in module._modules


def test_module_call():
    """Test that __call__ delegates to forward."""
    module = SimpleModule()
    assert module() == 123


def test_module_with_arguments():
    """Test module that accepts arguments in forward method."""
    module = ModuleWithArgs()
    result = module(10, text="hello")
    assert result == (20, "HELLO")


def test_module_with_default_arguments():
    """Test module using default argument values."""
    module = ModuleWithArgs()
    result = module(5)
    assert result == (10, "DEFAULT")


def test_forward_not_implemented():
    """Test that base Module class raises NotImplementedError for forward."""
    module = Module[Any]()
    with pytest.raises(NotImplementedError):
        module.forward()


def test_nested_modules():
    """Test nested module structure."""

    parent = CompoundModule()

    assert "child1" in parent._modules
    assert "child2" in parent._modules
    assert parent() == (123 + 123)


def test_module_type_annotation():
    """Test that type annotations are properly handled."""
    module: Module[int] = SimpleModule()
    result: int = module()
    assert isinstance(result, int)
    assert result == 123


def test_register_parameter():
    module = Module[Any]()
    param = Parameter([1.0])
    module.register_parameter("param", param)

    params = list(module.parameters())

    assert param in params


def test_parameters():
    model = SimpleModule()
    params = list(model.parameters())

    assert len(params) == 2
    assert model.param1 in params
    assert model.param2 in params


def test_named_parameters():
    model = SimpleModule()
    named_params = dict(model.named_parameters())

    assert len(named_params) == 2
    assert "param1" in named_params
    assert "param2" in named_params
    assert named_params["param1"] == model.param1
    assert named_params["param2"] == model.param2


def test_named_parameters_compound_module():
    model = CompoundModule()
    named_params = dict(model.named_parameters())

    # Verify the total number of parameters
    assert len(named_params) == 6

    # Verify the top-level parameters
    assert "param1" in named_params
    assert "param2" in named_params
    assert named_params["param1"] == model.param1
    assert named_params["param2"] == model.param2

    # Verify the child module parameters
    # - Child 1
    assert "child1.param1" in named_params
    assert "child1.param2" in named_params
    assert named_params["child1.param1"] == model.child1.param1
    assert named_params["child1.param2"] == model.child1.param2
    # - Child 2
    assert "child2.param1" in named_params
    assert "child2.param2" in named_params
    assert named_params["child2.param1"] == model.child2.param1
    assert named_params["child2.param2"] == model.child2.param2


def test_named_parameters_compound_module_no_recurse():
    model = CompoundModule()
    named_params = dict(model.named_parameters(recurse=False))

    # Verify the total number of parameters
    assert len(named_params) == 2

    # Verify the top-level parameters
    assert "param1" in named_params
    assert "param2" in named_params
    assert named_params["param1"] == model.param1
    assert named_params["param2"] == model.param2
