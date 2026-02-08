from typing import override

import numpy as np

from microtorch.nn.modules.module import Module
from microtorch.nn.modules.parameter import Parameter
from microtorch.tensor import Tensor, functional as F


class Conv2d(Module[Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng()
        self.weight = Parameter(
            rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)),
            requires_grad=True,
        )
        self.bias = (
            Parameter(
                np.zeros((out_channels,)),
                requires_grad=True,
            )
            if bias
            else None
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @override
    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)
