import numpy as np
import pytest

from microtorch.nn import Linear
from microtorch.tensor import Tensor, functional as F

# pyright: reportPrivateUsage=false


def test_linear_initialization():
    linear = Linear(3, 2)
    assert linear.weight.shape == (3, 2)
    assert linear.bias.shape == (2,)
    assert linear.weight.requires_grad
    assert linear.bias.requires_grad


def test_linear_forward():
    linear = Linear(3, 2)
    x = Tensor(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
    )
    output = linear(x)
    assert output.shape == (2, 2)
    expected = x @ linear.weight + linear.bias
    np.testing.assert_array_almost_equal(output._data, expected._data)


def test_linear_gradient():
    linear = Linear(3, 2)
    x = Tensor(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
    )
    output = linear(x)
    F.sum(output).backward()
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None
    assert linear.weight.grad.shape == (3, 2)
    assert linear.bias.grad.shape == (2,)


def test_linear_batch_input():
    batch_size = 32
    linear = Linear(10, 5)
    x = Tensor(np.random.randn(batch_size, 10))
    output = linear(x)
    assert output.shape == (batch_size, 5)


def test_linear_invalid_input():
    linear = Linear(3, 2)
    with pytest.raises(ValueError):
        linear(Tensor(np.random.randn(4)))  # Wrong dimensions
