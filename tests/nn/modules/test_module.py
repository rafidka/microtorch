from typing import Any

import pytest

from microtorch.nn import Module

# pyright: reportPrivateUsage=false


class SimpleModule(Module[int]):
    """A simple test module that returns a constant value."""

    def forward(self) -> int:
        return 123


class ModuleWithArgs(Module[tuple[int, str]]):
    """A test module that processes args and kwargs."""

    def forward(self, x: int, text: str = "default") -> tuple[int, str]:
        return (x * 2, text.upper())


class ParentModule(Module[int]):
    def __init__(self):
        super().__init__()
        self.child1 = SimpleModule()
        self.child2 = SimpleModule()

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

    parent = ParentModule()

    assert "child1" in parent._modules
    assert "child2" in parent._modules
    assert parent() == (123 + 123)


def test_module_type_annotation():
    """Test that type annotations are properly handled."""
    module: Module[int] = SimpleModule()
    result: int = module()
    assert isinstance(result, int)
    assert result == 123
