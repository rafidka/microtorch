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
            np.random.randn(in_features, out_features),
            requires_grad=True,
        )
        self.bias = Tensor(
            np.zeros(out_features),
            requires_grad=True,
        )
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input tensor to have shape (..., {self.in_features}), "
                f"but got {input.shape}."
            )
        input_shaped = input.reshape((-1, self.in_features))

        output_shaped = input_shaped @ self.weight + self.bias

        output = output_shaped.reshape(input.shape[:-1] + (self.out_features,))

        return output
