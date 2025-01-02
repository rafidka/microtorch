from microtorch.tensor import Tensor, functional as F

from .module import Module


class ReLU(Module[Tensor]):
    """Applies the rectified linear unit function element-wise."""

    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)


class Softmax(Module[Tensor]):
    """Applies the softmax function to an input tensor."""

    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        exps = F.exp(x - F.max(x, axis=1, keepdims=True))
        s = F.sum(exps, axis=1, keepdims=True)
        return exps / s
