import numpy as np

from microtorch.tensor import Tensor

from .module import Module


class Linear(Module[Tensor]):
    """
    A linear transformation module.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Tensor(
            np.random.randn(in_features, out_features), requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias
