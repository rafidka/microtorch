import numpy as np
import pytest

from microtorch.nn.modules.loss import CrossEntropyLoss
from microtorch.tensor import Tensor

# pyright: reportPrivateUsage=false


DEFAULT_ATOL = 1e-10


def test_cross_entropy_loss():
    # Create a CrossEntropyLoss instance
    criterion = CrossEntropyLoss()

    logits = Tensor(
        np.array(
            [
                [2.0, 1.0, 0.1],
                [0.5, 2.5, 1.0],
            ]
        ),
        requires_grad=True,
    )
    target = Tensor(
        np.array(
            [
                0,
                2,
            ]
        ),
        requires_grad=False,
    )

    # Compute the loss
    loss = criterion.forward(logits, target)

    # Assert that the computed loss is close to the expected loss
    assert np.isclose(loss._data.item(), 1.1116928642534902, atol=DEFAULT_ATOL)


def test_cross_entropy_loss_shape_mismatch():
    # Create a CrossEntropyLoss instance
    criterion = CrossEntropyLoss()

    # Define input tensor (logits) and target tensor (class indices) with shape mismatch
    input = Tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    target = Tensor(
        [
            2,
            1,
            0,
        ]
    )

    # Expect an exception due to shape mismatch
    with pytest.raises(
        ValueError,
        match="Expected logits and target to have the same batch size, but got 2 and 3.",
    ):
        criterion.forward(input, target)
