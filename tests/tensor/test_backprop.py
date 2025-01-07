import numpy as np
import pytest

from microtorch.tensor import Tensor
from microtorch.tensor.backprop import backward


@pytest.fixture
def x() -> Tensor:
    """Set up test cases."""
    return Tensor([2.0], requires_grad=True)


@pytest.fixture
def y() -> Tensor:
    return Tensor([3.0], requires_grad=True)


def test_simple_backward(x: Tensor, y: Tensor) -> None:
    """Test backpropagation for a simple computation: z = x * y."""
    z = x * y
    backward(z)

    assert x.grad is not None
    assert x.grad.item() == 3.0  # dz/dx = y

    assert y.grad is not None
    assert y.grad.item() == 2.0  # dz/dy = x


def test_backward_chain(x: Tensor, y: Tensor) -> None:
    """Test backpropagation through a chain of operations."""
    # Compute w = (x * y) + x
    w = (x * y) + x
    backward(w)

    # dw/dx = y + 1
    assert x.grad is not None
    assert x.grad.item() == 4.0

    # dw/dy = x
    assert y.grad is not None
    assert y.grad.item() == 2.0


def test_backward_branching(x: Tensor, y: Tensor) -> None:
    """Test backpropagation with branching computations."""
    # Create branching computation: x * x + x * y
    z = x * x + x * y
    backward(z)

    # dz/dx = 2x + y
    assert x.grad is not None
    assert x.grad.item() == 7.0  # 2 * 2 + 3

    # dz/dy = x
    assert y.grad is not None
    assert y.grad.item() == 2.0  # x


def test_backward_no_grad(x: Tensor, y: Tensor) -> None:
    """Test that backward fails on tensors with requires_grad=False."""
    x = Tensor([1.0], requires_grad=False)
    with pytest.raises(Exception):
        backward(x)


def test_backward_non_scalar(x: Tensor, y: Tensor) -> None:
    """Test that backward fails on non-scalar tensors."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(Exception):
        backward(x)


def test_complex_graph(x: Tensor, y: Tensor) -> None:
    """Test backpropagation through a more complex computation graph."""
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    z = Tensor([4.0], requires_grad=True)

    # Compute: w = (x * y * z) + (x * y) + x
    w = (x * y * z) + (x * y) + x
    backward(w)

    # dw/dx = y * z + y + 1
    assert x.grad is not None
    assert x.grad.item() == 16.0  # 3 * 4 + 3 + 1

    # dw/dy = x * z + x
    assert y.grad is not None
    assert y.grad.item() == 10.0  # 2 * 4 + 2

    # dw/dz = x * y
    assert z.grad is not None
    assert z.grad.item() == 6.0  # 2 * 3


def test_topological_order(x: Tensor, y: Tensor) -> None:
    """Test correct topological ordering during backpropagation."""
    # Create a graph where order matters
    x = Tensor([2.0], requires_grad=True)
    y = x * x  # y = x^2
    z = y * x  # z = x^3
    backward(z)

    # dz/dx = 3x^2
    expected_grad = 3 * (2.0**2)
    assert x.grad is not None
    assert x.grad.item() == expected_grad


def test_multiple_backward_calls() -> None:
    """Test that gradients accumulate correctly with multiple backward calls."""
    x = Tensor([2.0], requires_grad=True)
    y = x * x

    # First backward pass
    backward(y)
    assert x.grad is not None
    first_grad = x.grad.copy()

    # Second backward pass should accumulate gradients
    backward(y)
    np.testing.assert_array_equal(x.grad, 2 * first_grad)


def test_zero_grad() -> None:
    """Test that gradients are initialized correctly for leaf tensors."""
    x = Tensor([2.0], requires_grad=True)
    y = x * x

    # Before backward
    assert x.grad is not None
    assert x.grad.item() == 0.0

    backward(y)

    # After backward
    assert x.grad is not None
    assert x.grad.item() == 4.0
