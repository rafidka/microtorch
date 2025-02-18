import numpy as np
import pytest

from microtorch.nn.modules.parameter import Parameter
from microtorch.optim.optimizer import Optimizer

# pyright: reportPrivateUsage=false


class MockOptimizer(Optimizer):
    """Mock optimizer for testing purposes that simply scales gradients."""

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param._data -= self.lr * param.grad


@pytest.fixture
def parameters():
    """Fixture to create a list of parameters."""
    param1 = Parameter(np.array([1.0, 2.0, 3.0]))
    param2 = Parameter(np.array([[4.0, 5.0], [6.0, 7.0]]))
    return [param1, param2]


def test_optimizer_cannot_instantiate():
    """Ensure the base Optimizer class cannot be instantiated."""
    with pytest.raises(TypeError):
        Optimizer(parameters=[], lr=0.01)  # type: ignore


def test_zero_grad(parameters: list[Parameter]):
    """Test that zero_grad correctly resets gradients."""
    optimizer = MockOptimizer(parameters, lr=0.1)

    # Assign some gradients
    parameters[0].grad = np.array([0.1, 0.2, 0.3])
    parameters[1].grad = np.array([[0.4, 0.5], [0.6, 0.7]])

    # Ensure gradients are assigned
    assert parameters[0].grad is not None
    assert parameters[1].grad is not None

    optimizer.zero_grad()

    # Check that gradients are now zero
    np.testing.assert_array_equal(
        parameters[0].grad, np.zeros_like(parameters[0]._data)
    )
    np.testing.assert_array_equal(
        parameters[1].grad, np.zeros_like(parameters[1]._data)
    )
