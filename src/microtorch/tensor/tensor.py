# Python imports
from collections.abc import Callable
from typing import Any

# 3rd-party imports
import numpy as np

# Local imports
from . import backprop, functional as F


class Tensor:
    """
    A class used to represent a Tensor.

    Attributes:
    -----------
    data : np.ndarray
        The data of the tensor.
    requires_grad : bool
        Whether the tensor requires gradient computation.
    grad : np.ndarray or None
        The gradient of the tensor, if it requires gradient computation.
    """

    def __init__(
        self, data: np.ndarray[Any, Any] | list[Any], requires_grad: bool = False
    ):
        # `data` is a single float or list of floats representing the tensor
        self._data: np.ndarray[Any, Any] = (
            data if isinstance(data, np.ndarray) else np.array(data)
        )
        self.requires_grad = requires_grad
        self.grad = None if not requires_grad else np.zeros(self._data.shape)

        # Protected attributes used for autograd.
        self._backward: Callable[[], None] | None = None
        self._is_leaf = True
        self._op = ""
        self._prev: list[Tensor] = []
        self._topo_order = 1

    def _move(self, other: "Tensor"):
        self._data = other._data
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self._backward = other._backward
        self._is_leaf = other._is_leaf
        self._op = other._op
        self._prev = other._prev
        self._topo_order = other._is_leaf

    def clone(self) -> "Tensor":
        # TODO This function should be differentiable just like pytorch's.
        return Tensor(self._data.copy(), self.requires_grad)

    def numpy(self) -> np.ndarray[Any, Any]:
        # Return a copy of the data rather than the data itself (to avoid modifying the
        # original data)
        return self._data.copy()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    def __len__(self) -> int:
        return len(self._data)

    def __add__(self, other: "Tensor"):
        return F.add(self, other)

    def __iadd__(self, other: "Tensor"):
        self._move(F.add(self, other))
        return self

    def __sub__(self, other: "Tensor"):
        return F.sub(self, other)

    def __isub__(self, other: "Tensor"):
        self._move(F.sub(self, other))
        return self

    def __neg__(self):
        return F.neg(self)

    def __mul__(self, other: "Tensor"):
        return F.mul(self, other)

    def __imul__(self, other: "Tensor"):
        self._move(F.mul(self, other))
        return self

    def __matmul__(self, other: "Tensor"):
        return F.matmul(self, other)

    def __truediv__(self, other: "Tensor"):
        return F.div(self, other)

    def __itruediv__(self, other: "Tensor"):
        self._move(F.div(self, other))
        return self

    def reshape(self, *shape: int | tuple[int, ...]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
            return F.reshape(self, shape)
        else:
            return F.reshape(self, shape)  # type: ignore

    def backward(self):
        return backprop.backward(self)

    def item(self):
        if self._data.size != 1:
            raise ValueError(
                "only one element tensors can be converted to Python scalars"
            )
        return self._data.item()

    def __repr__(self):
        return f"Tensor(data={self._data}, shape={self.shape}, requires_grad={self.requires_grad}, grad={self.grad})"
