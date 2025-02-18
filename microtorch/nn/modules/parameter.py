from typing import TYPE_CHECKING, Any

import numpy as np

from microtorch.tensor import Tensor


class Parameter(Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.
    """

    def __init__(
        self, data: np.ndarray[Any, Any] | list[Any], requires_grad: bool = False
    ):
        super().__init__(data, requires_grad)
        if TYPE_CHECKING:
            # Let type checkers know that grad can be None or np.ndarray (for some
            # reason, VSCode complains when I try to set 'grad' to None in test_sgd.py)
            self.grad: np.ndarray[Any, Any] | None

    def zero_grad(self):
        """Clears the gradients of the parameter."""
        if self.grad is None:
            return
        self.grad = np.zeros(self._data.shape)
