from microtorch.nn.modules.module import Module
from microtorch.tensor import Tensor, functional as F


class CrossEntropyLoss(Module[Tensor]):
    """
    A module that computes the cross-entropy loss between logits and target.

    This is often used for multi-class classification problems. It expects:
      - input of shape (N, C) where N is batch size, C is number of classes
      - target of shape (N,) with class indices in [0, C-1]
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target)
