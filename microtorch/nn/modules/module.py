# 3rd party imports
from typing import Any

# Local imports


class Module[T]:
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        self._modules: dict[str, Module[Any]] = {}

    def add_module(self, name: str, module: "Module[T]"):
        self._modules[name] = module

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Module):
            self.add_module(name, value)  # type: ignore
        super().__setattr__(name, value)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> T:
        raise NotImplementedError
