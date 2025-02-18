import numpy as np
import pytest

from microtorch.nn.modules.parameter import Parameter
from microtorch.optim.sgd import SGD

# pyright: reportPrivateUsage=false


@pytest.fixture
def parameters():
    """Fixture to create a list of parameters."""
    param1 = Parameter(np.array([1.0, 2.0, 3.0]))
    param2 = Parameter(np.array([[4.0, 5.0], [6.0, 7.0]]))
    return [param1, param2]


def test_sgd_basic_step(parameters: list[Parameter]):
    """Test if SGD updates parameters correctly."""
    optimizer = SGD(parameters, lr=0.1)

    # Assign gradients
    parameters[0].grad = np.array([0.1, 0.2, 0.3])
    parameters[1].grad = np.array([[0.4, 0.5], [0.6, 0.7]])

    # Perform optimization step
    optimizer.step()

    # Expected updated values
    expected_0 = np.array([0.99, 1.98, 2.97])  # 1.0 - 0.1 * 0.1, etc.
    expected_1 = np.array([[3.96, 4.95], [5.94, 6.93]])

    np.testing.assert_almost_equal(parameters[0]._data, expected_0)
    np.testing.assert_almost_equal(parameters[1]._data, expected_1)


def test_sgd_with_zero_gradient(parameters: list[Parameter]):
    """Test if SGD correctly handles zero gradients."""
    optimizer = SGD(parameters, lr=0.1)

    # Set zero gradients
    parameters[0].zero_grad()
    parameters[1].zero_grad()

    original_0 = parameters[0]._data.copy()
    original_1 = parameters[1]._data.copy()

    optimizer.step()

    np.testing.assert_array_equal(parameters[0]._data, original_0)
    np.testing.assert_array_equal(parameters[1]._data, original_1)


def test_sgd_no_gradients(parameters: list[Parameter]):
    """Test if SGD skips parameters with no gradients."""
    optimizer = SGD(parameters, lr=0.1)

    # Leave gradients as None
    parameters[0].grad = None
    parameters[1].grad = None

    original_0 = parameters[0]._data.copy()
    original_1 = parameters[1]._data.copy()

    optimizer.step()

    np.testing.assert_array_equal(parameters[0]._data, original_0)
    np.testing.assert_array_equal(parameters[1]._data, original_1)


def test_sgd_different_learning_rates(parameters: list[Parameter]):
    """Test if different learning rates affect parameter updates correctly."""
    optimizer = SGD(parameters, lr=0.5)

    parameters[0].grad = np.array([0.2, 0.4, 0.6])
    parameters[1].grad = np.array([[0.8, 1.0], [1.2, 1.4]])

    optimizer.step()

    expected_0 = np.array([0.9, 1.8, 2.7])  # 1.0 - 0.5 * 0.2, etc.
    expected_1 = np.array([[3.6, 4.5], [5.4, 6.3]])

    np.testing.assert_almost_equal(parameters[0]._data, expected_0)
    np.testing.assert_almost_equal(parameters[1]._data, expected_1)
