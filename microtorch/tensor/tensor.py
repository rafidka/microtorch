# 3rd-party imports
import numpy as np

# Local imports
from . import backprop, functional as F


class Tensor:
    def __init__(self, data: np.ndarray, requires_grad=False):
        # `data` is a single float or list of floats representing the tensor
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None if not requires_grad else np.zeros(self.data.shape)
        self._backward = None
        self._prev: list[Tensor] = []
        self._op = ""
        self.is_leaf = True
        self._topo_order = 1

    def __add__(self, other):
        return F.add(self, other)

    def __neg__(self):
        return F.neg(self)

    def __mul__(self, other):
        return F.mul(self, other)

    def __matmul__(self, other):
        return F.matmul(self, other)

    def backward(self):
        return backprop.backward(self)

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"
