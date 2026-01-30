from typing import override

from microtorch.tensor import Tensor, functional as F

from .module import Module


class ReLU(Module[Tensor]):
    """Applies the rectified linear unit function element-wise."""

    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)


class Softmax(Module[Tensor]):
    """Applies the softmax function to an input tensor."""

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    @override
    def forward(self, x: Tensor) -> Tensor:
        exps = F.exp(x - F.max(x, axis=self.dim, keepdims=True))
        s = F.sum(exps, axis=self.dim, keepdims=True)
        return exps / s
