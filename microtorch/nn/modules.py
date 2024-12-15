# 3rd party imports
from typing import Any
import numpy as np

# Local imports
from microtorch.tensor import Tensor, functional as F


class Module[T]:
    def __init__(self):
        self._modules: dict[str, "Module[T]"] = {}

    def add_module(self, name: str, module: "Module[T]"):
        self._modules[name] = module

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Module):
            self.add_module(name, value)  # type: ignore
        super(Module, self).__setattr__(name, value)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> T:
        raise NotImplementedError


class Linear(Module[Tensor]):
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()
        self.weight = Tensor(
            np.random.randn(in_features, out_features), requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


class Softmax(Module[Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        exps = F.exp(x - F.max(x, axis=1, keepdims=True))
        s = F.sum(exps, axis=1, keepdims=True)
        return exps / s
