"""Edge case tests for tensor operations.

These tests verify the framework handles edge cases gracefully,
including empty tensors, NaN/Inf values, and numerical edge cases.
"""

import numpy as np
import pytest

from microtorch.nn import Linear, Softmax
from microtorch.tensor import Tensor, functional as F
from microtorch.utils.data import Dataset


class TestEmptyTensors:
    """Tests for empty tensor operations."""

    def test_empty_tensor_creation(self):
        """Test creating tensors with zero elements."""
        # 1D empty tensor
        t1 = Tensor(np.array([]))
        assert t1.shape == (0,)
        assert len(t1) == 0

        # 2D empty tensor
        t2 = Tensor(np.zeros((0, 5)))
        assert t2.shape == (0, 5)

        # Another 2D empty
        t3 = Tensor(np.zeros((3, 0)))
        assert t3.shape == (3, 0)

    def test_empty_tensor_sum(self):
        """Test sum of empty tensor."""
        t = Tensor(np.array([]))
        result = F.sum(t)
        assert result.item() == 0.0

    def test_empty_tensor_reshape(self):
        """Test reshaping empty tensors."""
        t = Tensor(np.zeros((0, 5)))
        reshaped = t.reshape((5, 0))
        assert reshaped.shape == (5, 0)


class TestScalarTensors:
    """Tests for 0-dimensional (scalar) tensors."""

    def test_scalar_tensor_creation(self):
        """Test creating scalar tensors."""
        t = Tensor(np.array(5.0))
        assert t.shape == ()
        assert t.ndim == 0
        assert t.item() == 5.0

    def test_scalar_tensor_arithmetic(self):
        """Test arithmetic with scalar tensors."""
        a = Tensor(np.array(3.0))
        b = Tensor(np.array(4.0))

        assert (a + b).item() == 7.0
        assert (a - b).item() == -1.0
        assert (a * b).item() == 12.0
        assert (a / b).item() == 0.75

    def test_scalar_tensor_backward(self):
        """Test backward through scalar tensor."""
        a = Tensor(np.array(3.0), requires_grad=True)
        b = Tensor(np.array(4.0), requires_grad=True)

        c = a * b
        c.backward()

        assert a.grad is not None
        assert a.grad.item() == 4.0  # dc/da = b
        assert b.grad is not None
        assert b.grad.item() == 3.0  # dc/db = a


class TestNaNHandling:
    """Tests for NaN value handling."""

    def test_nan_propagation_add(self):
        """Test that NaN propagates through addition."""
        a = Tensor(np.array([1.0, np.nan, 3.0]))
        b = Tensor(np.array([1.0, 2.0, 3.0]))
        result = a + b
        assert np.isnan(result.numpy()[1])
        assert result.numpy()[0] == 2.0
        assert result.numpy()[2] == 6.0

    def test_nan_propagation_mul(self):
        """Test that NaN propagates through multiplication."""
        a = Tensor(np.array([1.0, np.nan, 3.0]))
        b = Tensor(np.array([2.0, 2.0, 2.0]))
        result = a * b
        assert np.isnan(result.numpy()[1])

    def test_nan_in_sum(self):
        """Test sum with NaN values."""
        a = Tensor(np.array([1.0, np.nan, 3.0]))
        result = F.sum(a)
        assert np.isnan(result.item())

    def test_nan_backward(self):
        """Test backward with NaN in computation."""
        a = Tensor(np.array([1.0, np.nan]), requires_grad=True)
        b = F.sum(a)
        b.backward()
        # Gradient should still be computed (as 1s)
        assert a.grad is not None


class TestInfHandling:
    """Tests for infinity value handling."""

    def test_inf_from_exp_overflow(self):
        """Test that exp of large values produces inf."""
        a = Tensor(np.array([1.0, 1000.0]))  # 1000 will overflow
        result = F.exp(a)
        assert np.isinf(result.numpy()[1])
        assert np.isfinite(result.numpy()[0])

    def test_inf_propagation(self):
        """Test that inf propagates through operations."""
        a = Tensor(np.array([np.inf, 1.0]))
        b = Tensor(np.array([1.0, 1.0]))
        result = a + b
        assert np.isinf(result.numpy()[0])
        assert result.numpy()[1] == 2.0

    def test_inf_times_zero(self):
        """Test inf * 0 = nan."""
        a = Tensor(np.array([np.inf]))
        b = Tensor(np.array([0.0]))
        result = a * b
        assert np.isnan(result.item())

    def test_div_by_zero(self):
        """Test division by zero produces inf."""
        a = Tensor(np.array([1.0, -1.0, 0.0]))
        b = Tensor(np.array([0.0, 0.0, 0.0]))
        result = a / b
        assert np.isinf(result.numpy()[0])  # 1/0 = inf
        assert np.isinf(result.numpy()[1])  # -1/0 = -inf
        assert np.isnan(result.numpy()[2])  # 0/0 = nan


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_softmax_large_values(self):
        """Test softmax is stable with large values."""
        # Without stability tricks, this would overflow
        softmax = Softmax(dim=0)
        a = Tensor(np.array([1000.0, 1001.0, 1002.0]))
        result = softmax(a)

        # Result should be valid probabilities
        assert np.all(np.isfinite(result.numpy()))
        assert np.all(result.numpy() >= 0)
        assert np.isclose(F.sum(result).item(), 1.0)

    def test_softmax_very_negative(self):
        """Test softmax with very negative values."""
        softmax = Softmax(dim=0)
        a = Tensor(np.array([-1000.0, -1001.0, -1002.0]))
        result = softmax(a)

        assert np.all(np.isfinite(result.numpy()))
        assert np.all(result.numpy() >= 0)
        assert np.isclose(F.sum(result).item(), 1.0)

    def test_cross_entropy_stability(self):
        """Test cross entropy is stable with extreme logits."""
        logits = Tensor(np.array([[100.0, 0.0, -100.0]]), requires_grad=True)
        target = Tensor(np.array([0]))

        loss = F.cross_entropy(logits, target)

        # Loss should be finite and close to 0 (correct class has high logit)
        assert np.isfinite(loss.item())
        assert loss.item() < 1.0  # Should be very low loss


class TestMaxTiedValues:
    """Tests for max operation with tied values."""

    def test_max_tied_gradient_distribution(self):
        """Test gradient is distributed among tied max values."""
        # All values equal
        a = Tensor(np.array([5.0, 5.0, 5.0, 5.0]), requires_grad=True)
        result = F.max(a)
        result.backward()

        # Gradient should be distributed evenly: 1/4 = 0.25 each
        assert a.grad is not None
        np.testing.assert_array_almost_equal(a.grad, np.array([0.25, 0.25, 0.25, 0.25]))

    def test_max_two_tied(self):
        """Test max with two tied values."""
        a = Tensor(np.array([1.0, 5.0, 3.0, 5.0]), requires_grad=True)
        result = F.max(a)
        result.backward()

        # Only positions 1 and 3 are max, so they should share gradient
        assert a.grad is not None
        expected = np.array([0.0, 0.5, 0.0, 0.5])
        np.testing.assert_array_almost_equal(a.grad, expected)

    def test_max_axis_tied(self):
        """Test max along axis with tied values."""
        a = Tensor(np.array([[5.0, 5.0], [3.0, 5.0]]), requires_grad=True)
        result = F.max(a, axis=1)
        F.sum(result).backward()

        assert a.grad is not None
        # Row 0: both are max, split gradient
        # Row 1: only col 1 is max
        expected = np.array([[0.5, 0.5], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(a.grad, expected)


class TestReshapeWithNegativeOne:
    """Tests for reshape with -1 dimension."""

    def test_reshape_infer_first_dim(self):
        """Test reshape inferring first dimension."""
        a = Tensor(np.arange(12))
        b = a.reshape((-1, 3))
        assert b.shape == (4, 3)

    def test_reshape_infer_last_dim(self):
        """Test reshape inferring last dimension."""
        a = Tensor(np.arange(12))
        b = a.reshape((3, -1))
        assert b.shape == (3, 4)

    def test_reshape_infer_middle_dim(self):
        """Test reshape inferring middle dimension."""
        a = Tensor(np.arange(24))
        b = a.reshape((2, -1, 3))
        assert b.shape == (2, 4, 3)


class TestDatasetBaseClass:
    """Tests for Dataset base class."""

    def test_dataset_getitem_not_implemented(self):
        """Test that base Dataset raises NotImplementedError for __getitem__."""
        dataset = Dataset()
        with pytest.raises(NotImplementedError):
            _ = dataset[0]

    def test_dataset_len_not_implemented(self):
        """Test that base Dataset raises NotImplementedError for __len__."""
        dataset = Dataset()
        with pytest.raises(NotImplementedError):
            len(dataset)


class TestVerySmallGradients:
    """Tests for handling very small gradient values."""

    def test_small_gradient_not_zero(self):
        """Test that very small gradients are preserved, not zeroed."""
        # Create a situation with small gradients
        a = Tensor(np.array([1e-10]), requires_grad=True)
        b = a * Tensor(np.array([1e-10]))
        c = F.sum(b)
        c.backward()

        assert a.grad is not None
        assert a.grad[0] != 0.0
        assert a.grad[0] == pytest.approx(1e-10, rel=1e-5)


class TestLargeValueOperations:
    """Tests for operations with large values."""

    def test_add_large_values(self):
        """Test addition with large values."""
        a = Tensor(np.array([1e15, 1e15]))
        b = Tensor(np.array([1e15, 1e15]))
        result = a + b
        np.testing.assert_array_equal(result.numpy(), np.array([2e15, 2e15]))

    def test_matmul_large_values(self):
        """Test matmul with moderately large values."""
        a = Tensor(np.array([[1e6, 1e6], [1e6, 1e6]]))
        b = Tensor(np.array([[1e6, 1e6], [1e6, 1e6]]))
        result = a @ b
        expected = np.array([[2e12, 2e12], [2e12, 2e12]])
        np.testing.assert_array_almost_equal(result.numpy(), expected)


class TestBroadcastingEdgeCases:
    """Tests for broadcasting edge cases."""

    def test_broadcast_scalar_to_matrix(self):
        """Test broadcasting a scalar to a matrix."""
        a = Tensor(np.array([[1, 2], [3, 4]]))
        b = Tensor(np.array([10]))
        result = a + b
        expected = np.array([[11, 12], [13, 14]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_broadcast_column_to_matrix(self):
        """Test broadcasting a column vector to a matrix."""
        a = Tensor(np.array([[1, 2], [3, 4]]))
        b = Tensor(np.array([[10], [20]]))
        result = a + b
        expected = np.array([[11, 12], [23, 24]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_broadcast_with_gradients(self):
        """Test broadcasting correctly accumulates gradients."""
        a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
        b = Tensor(np.array([10]), requires_grad=True)  # Broadcasts to all elements
        c = F.sum(a + b)
        c.backward()

        # a's gradient should be all 1s
        assert a.grad is not None
        np.testing.assert_array_equal(a.grad, np.ones((2, 2)))

        # b's gradient should be sum of all (4 elements)
        assert b.grad is not None
        assert b.grad[0] == 4.0
