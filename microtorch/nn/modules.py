# 3rd party imports
import numpy as np

# Local imports
from microtorch.tensor import Tensor, functional as F


class Module:
    def __init__(self):
        self._modules = {}

    def add_module(self, name, module):
        self._modules[name] = module

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.add_module(name, value)
        super(Module, self).__setattr__(name, value)

    def __getattr__(self, name):
        if name in self._modules:
            return self._modules[name]
        return super(Module, self).__getattr__(name)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weight = Tensor(
            np.random.randn(in_features, out_features), requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x):
        return x @ self.weight + self.bias


class Softmax(Module):
    def forward(self, x):
        exps = F.exp(x - F.max(x, axis=1, keepdims=True))
        return exps / sum(exps, axis=1, keepdims=True)
