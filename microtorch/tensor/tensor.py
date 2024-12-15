# 3rd-party imports
from typing import Any, Callable, Optional
import numpy as np

# Local imports
from . import backprop, functional as F


class Tensor:
    def __init__(self, data: np.ndarray[Any, Any], requires_grad: bool = False):
        # `data` is a single float or list of floats representing the tensor
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None if not requires_grad else np.zeros(self.data.shape)

        # Protected attributes used for autograd.
        self._backward: Optional[Callable[[], None]] = None
        self._is_leaf = True
        self._op = ""
        self._prev: list[Tensor] = []
        self._topo_order = 1

    def _move(self, other: "Tensor") -> None:
        self.data = other.data
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self._backward = other._backward
        self._is_leaf = other._is_leaf
        self._op = other._op
        self._prev = other._prev
        self._topo_order = other._is_leaf

    def __add__(self, other: "Tensor"):
        return F.add(self, other)

    def __iadd__(self, other: "Tensor"):
        self._move(F.add(self, other))

    def __sub__(self, other: "Tensor"):
        return F.add(self, other)

    def __isub__(self, other: "Tensor"):
        self._move(F.add(self, other))

    def __neg__(self):
        return F.neg(self)

    def __mul__(self, other: "Tensor"):
        return F.mul(self, other)

    def __imul__(self, other: "Tensor"):
        self._move(F.mul(self, other))

    def __matmul__(self, other: "Tensor"):
        return F.matmul(self, other)

    def __truediv__(self, other: "Tensor"):
        return F.div(self, other)

    def __itruediv__(self, other: "Tensor"):
        self._move(F.div(self, other))

    def backward(self):
        return backprop.backward(self)

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"
