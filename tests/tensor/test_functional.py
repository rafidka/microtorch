import numpy as np
import pytest

from microtorch.tensor import functional as F, tensor

# pyright: reportPrivateUsage=false

DEFAULT_ATOL = 1e-10


def test_add_with_grads():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=True)
    result = F.add(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([4, 6]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([1, 1]))


def test_add_without_grads():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=False)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=False)
    result = F.add(a, b)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0, 0]), requires_grad=True)
    F.sum(result + dummy).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([4, 6]))

    # Check the gradients
    assert a.grad is None
    assert b.grad is None


def test_add_with_lhs_grad():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=False)
    result = F.add(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([4, 6]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    assert b.grad is None


def test_add_with_rhs_grad():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=False)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=True)
    result = F.add(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([4, 6]))

    # Check the gradients
    assert a.grad is None
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([1, 1]))


def test_add_with_broadcasting():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3]), requires_grad=True)
    result = F.add(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([4, 5]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([2]))


def test_sub_with_grads():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=True)
    result = F.sub(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([-2, -2]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([-1, -1]))


def test_sub_without_grads():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=False)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=False)
    result = F.sub(a, b)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0, 0]), requires_grad=True)
    F.sum(result + dummy).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([-2, -2]))

    # Check the gradients
    assert a.grad is None
    assert b.grad is None


def test_sub_with_lhs_grad():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=False)
    result = F.sub(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([-2, -2]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    assert b.grad is None


def test_sub_with_rhs_grad():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=False)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=True)
    result = F.sub(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([-2, -2]))

    # Check the gradients
    assert a.grad is None
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([-1, -1]))


def test_sub_with_broadcasting():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3]), requires_grad=True)
    result = F.sub(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([-2, -1]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([-2]))


def test_neg_with_grads():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    result = F.neg(a)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([-1, -2]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([-1, -1]))


def test_neg_without_grads():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=False)
    result = F.neg(a)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0, 0]), requires_grad=True)
    F.sum(result + dummy).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([-1, -2]))

    # Check the gradients
    assert a.grad is None


def test_mul_with_grads():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=True)
    result = F.mul(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([3, 8]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([3, 4]))
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([1, 2]))


def test_mul_without_grads():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=False)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=False)
    result = F.mul(a, b)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0, 0]), requires_grad=True)
    F.sum(result + dummy).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([3, 8]))

    # Check the gradients
    assert a.grad is None
    assert b.grad is None


def test_mul_with_lhs_grad():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=False)
    result = F.mul(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([3, 8]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([3, 4]))
    assert b.grad is None


def test_mul_with_rhs_grad():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=False)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=True)
    result = F.mul(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([3, 8]))

    # Check the gradients
    assert a.grad is None
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([1, 2]))


def test_mul_with_broadcasting():
    a = tensor.Tensor(np.array([1, 3]), requires_grad=True)
    b = tensor.Tensor(np.array([7]), requires_grad=True)
    result = F.mul(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([1 * 7, 3 * 7]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([7, 7]))
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([1 + 3]))


def test_matmul_with_grads():
    a = tensor.Tensor(np.array([[1, 3], [3, 10]]), requires_grad=True)
    b = tensor.Tensor(np.array([[5, -2], [10, 20]]), requires_grad=True)
    result = F.matmul(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([[35, 58], [115, 194]]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([[3, 30], [3, 30]]))
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([[4, 4], [13, 13]]))


def test_matmul_without_grads():
    a = tensor.Tensor(np.array([[1, 3], [3, 10]]), requires_grad=False)
    b = tensor.Tensor(np.array([[5, -2], [10, 20]]), requires_grad=False)
    result = F.matmul(a, b)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0, 0]), requires_grad=True)
    F.sum(result + dummy).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([[35, 58], [115, 194]]))

    # Check the gradients
    assert a.grad is None
    assert b.grad is None


def test_matmul_with_lhs_grad():
    a = tensor.Tensor(np.array([[1, 3], [3, 10]]), requires_grad=True)
    b = tensor.Tensor(np.array([[5, -2], [10, 20]]), requires_grad=False)
    result = F.matmul(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([[35, 58], [115, 194]]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([[3, 30], [3, 30]]))
    assert b.grad is None


def test_matmul_with_rhs_grad():
    a = tensor.Tensor(np.array([[1, 3], [3, 10]]), requires_grad=False)
    b = tensor.Tensor(np.array([[5, -2], [10, 20]]), requires_grad=True)
    result = F.matmul(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([[35, 58], [115, 194]]))

    # Check the gradients
    assert a.grad is None
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([[4, 4], [13, 13]]))


def test_div_with_grads():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=True)
    result = F.div(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([1 / 3, 2 / 4]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1 / 3, 1 / 4]))
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([-1 / 9, -1 / 8]))


def test_div_without_grads():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=False)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=False)
    result = F.div(a, b)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0, 0]), requires_grad=True)
    F.sum(result + dummy).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([1 / 3, 2 / 4]))

    # Check the gradients
    assert a.grad is None
    assert b.grad is None


def test_div_with_lhs_grad():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=False)
    result = F.div(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([1 / 3, 2 / 4]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1 / 3, 1 / 4]))
    assert b.grad is None


def test_div_with_rhs_grad():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=False)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=True)
    result = F.div(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([1 / 3, 2 / 4]))

    # Check the gradients
    assert a.grad is None
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([-1 / 9, -1 / 8]))


def test_div_with_broadcasting():
    a = tensor.Tensor(np.array([1, 3]), requires_grad=True)
    b = tensor.Tensor(np.array([7]), requires_grad=True)
    result = F.div(a, b)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([1 / 7, 3 / 7]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1 / 7, 1 / 7]))
    assert b.grad is not None
    assert np.array_equal(b.grad, np.array([-1 / 49 - 3 / 49]))


def test_sin_with_grads():
    a = tensor.Tensor(np.array([0, np.pi / 2]), requires_grad=True)
    result = F.sin(a)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([0, 1]))

    # Check the gradients
    assert a.grad is not None
    assert np.allclose(a.grad, np.array([1, 0]))


def test_sin_without_grads():
    a = tensor.Tensor(np.array([0, np.pi / 2]), requires_grad=False)
    result = F.sin(a)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0, 0]), requires_grad=True)
    F.sum(result + dummy).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([0, 1]))

    # Check the gradients
    assert a.grad is None


def test_cos_with_grads():
    a = tensor.Tensor(np.array([0, np.pi / 2]), requires_grad=True)
    result = F.cos(a)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    np.testing.assert_almost_equal(result._data, np.array([1, 0]))

    # Check the gradients
    assert a.grad is not None
    assert np.allclose(a.grad, np.array([0, -1]))


def test_cos_without_grads():
    a = tensor.Tensor(np.array([0, np.pi / 2]), requires_grad=False)
    result = F.cos(a)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0, 0]), requires_grad=True)
    F.sum(result + dummy).backward()  # do a sum to be able to call backward()

    # Check the result
    np.testing.assert_almost_equal(result._data, np.array([1, 0]))

    # Check the gradients
    assert a.grad is None


def test_exp_with_grads():
    a = tensor.Tensor(np.array([0, 1]), requires_grad=True)
    result = F.exp(a)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    np.testing.assert_almost_equal(result._data, np.array([1, np.exp(1)]))

    # Check the gradients
    assert a.grad is not None
    assert np.allclose(a.grad, np.array([1, np.exp(1)]))


def test_exp_without_grads():
    a = tensor.Tensor(np.array([0, 1]), requires_grad=False)
    result = F.exp(a)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0, 0]), requires_grad=True)
    F.sum(result + dummy).backward()  # do a sum to be able to call backward()

    # Check the result
    np.testing.assert_almost_equal(result._data, np.array([1, np.exp(1)]))

    # Check the gradients
    assert a.grad is None


def test_sum_with_grads():
    a = tensor.Tensor(np.array([1, 2, 3, 4]), requires_grad=True)
    result = F.sum(a)

    result.backward()

    # Check the result
    assert result._data == 10

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1, 1, 1]))


def test_sum_without_grads():
    a = tensor.Tensor(np.array([1, 2, 3, 4]), requires_grad=False)
    result = F.sum(a)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0]), requires_grad=True)
    (result + dummy).backward()

    # Check the result
    assert result._data == 10

    # Check the gradients
    assert a.grad is None


def test_max_with_grads():
    a = tensor.Tensor(np.array([1, 2, 3, 4]), requires_grad=True)
    result = F.max(a)

    result.backward()

    # Check the result
    assert result._data == 4

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([0, 0, 0, 1]))


def test_max_without_grads():
    a = tensor.Tensor(np.array([1, 2, 3, 4]), requires_grad=False)
    result = F.max(a)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0]), requires_grad=True)
    (result + dummy).backward()

    # Check the result
    assert result._data == 4

    # Check the gradients
    assert a.grad is None


def test_relu_with_grads():
    a = tensor.Tensor(np.array([-1, 2, 3, -4]), requires_grad=True)
    result = F.relu(a)

    F.sum(result).backward()

    # Check the result
    assert np.array_equal(result._data, np.array([0, 2, 3, 0]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([0, 1, 1, 0]))


def test_relu_without_grads():
    a = tensor.Tensor(np.array([-1, 2, 3, -4]), requires_grad=False)
    result = F.relu(a)

    # Add a dummy differentiable tensor to be able to call backward()
    dummy = tensor.Tensor(np.array([0, 0, 0, 0]), requires_grad=True)
    F.sum(result + dummy).backward()

    # Check the result
    assert np.array_equal(result._data, np.array([0, 2, 3, 0]))

    # Check the gradients
    assert a.grad is None


def test_reshape():
    a = tensor.Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
    result = F.reshape(a, (1, 4))

    F.sum(result).backward()

    # Check the result
    assert np.array_equal(result._data, np.array([[1, 2, 3, 4]]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([[1, 1], [1, 1]]))


def test_reshape_backward_chain():
    a = tensor.Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
    result = F.reshape(a, (1, 4))
    result2 = F.reshape(result, (2, 2))

    F.sum(result2).backward()

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([[1, 1], [1, 1]]))


def test_reshape_exp_chain():
    a = tensor.Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
    result = F.reshape(a, (1, 4))
    result2 = F.exp(result)

    F.sum(result2).backward()

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.exp(a._data))


def test_reshape_sine_cosine_chain():
    a = tensor.Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
    result = F.reshape(a, (1, 4))
    result2 = F.sin(result)
    result3 = F.cos(result2)

    F.sum(result3).backward()

    # Check the gradients
    assert a.grad is not None
    assert np.allclose(a.grad, -np.sin(np.sin(a._data)) * np.cos(a._data))


def test_cross_entropy_basic_forward_single():
    """
    Test the forward pass of cross_entropy with a simple example.
    """
    logits = tensor.Tensor(
        np.array(
            [
                [2.0, 1.0, 0.1],
            ]
        ),
        requires_grad=True,
    )
    target = tensor.Tensor(
        np.array(
            [
                0,
            ]
        ),
        requires_grad=False,
    )

    loss = F.cross_entropy(logits, target)

    # Confirm the type, shape and size of the loss.
    assert (
        isinstance(loss._data, np.ndarray)
        and loss._data.shape == ()
        and loss._data.size == 1
    ), "Cross-entropy should return a numpy array with a single float element."

    # Confirm the value of the loss. The value was calculated manually using PyTorch.
    expected_loss = 0.417029947042465
    assert np.isclose(loss._data.item(), expected_loss, atol=DEFAULT_ATOL)


def test_cross_entropy_basic_forward_batch():
    """
    Test the forward pass of cross_entropy with a simple example.
    """
    logits = tensor.Tensor(
        np.array(
            [
                [2.0, 1.0, 0.1],
                [0.5, 2.5, 1.0],
                [10.0, -2.5, -50.123],
            ]
        ),
        requires_grad=True,
    )
    target = tensor.Tensor(
        np.array(
            [
                0,
                2,
                1,
            ]
        ),
        requires_grad=False,
    )

    loss = F.cross_entropy(logits, target)

    # Confirm the type, shape and size of the loss.
    assert (
        isinstance(loss._data, np.ndarray)
        and loss._data.shape == ()
        and loss._data.size == 1
    ), "Cross-entropy should return a numpy array with a single float element."

    # Confirm the value of the loss. The value was calculated manually using PyTorch.
    expected_loss = 4.907796382904053
    assert np.isclose(loss._data.item(), expected_loss, atol=1e-10)


def test_cross_entropy_backward():
    """
    Test the backward pass (gradient computation) of cross_entropy.
    We'll compare the computed gradients to a numerical approximation.
    """
    logits = tensor.Tensor(
        np.array(
            [
                [2.0, 1.0, 0.1],
                [0.5, 2.5, 1.0],
                [10.0, -2.5, -50.123],
            ]
        ),
        requires_grad=True,
    )
    target = tensor.Tensor(
        np.array(
            [
                0,
                2,
                1,
            ]
        ),
        requires_grad=False,
    )

    loss = F.cross_entropy(logits, target)

    loss.backward()
    assert logits.grad is not None
    expected_grad = np.array(
        [
            [-1.136662811040878e-01, 8.081099390983582e-02, 3.285530209541321e-02],
            [3.320788592100143e-02, 2.453749179840088e-01, -2.785828113555908e-01],
            [3.333321213722229e-01, -3.333320915699005e-01, 2.581008822323310e-27],
        ]
    )
    assert np.all(np.isclose(logits.grad, expected_grad, atol=DEFAULT_ATOL))

    # Assert the target does not have a gradient
    assert target.grad is None


def test_cross_entropy_no_grad():
    """
    Test cross_entropy when the input logits do not require gradients.
    """

    # Create tensors, but logits do not require grad
    logits = tensor.Tensor(
        np.array(
            [
                [2.0, 1.0, 0.1],
                [0.5, 2.5, 1.0],
            ]
        ),
        requires_grad=False,
    )
    target = tensor.Tensor(
        np.array(
            [
                0,
                2,
            ]
        ),
        requires_grad=False,
    )

    loss = F.cross_entropy(logits, target)

    with pytest.raises(ValueError):
        loss.backward()


def test_stack_basic():
    """Test basic stacking functionality along default axis."""
    a = tensor.Tensor([1, 2], requires_grad=True)
    b = tensor.Tensor([3, 4], requires_grad=True)

    result = F.stack([a, b])

    # Check the result
    assert result._data.shape == (2, 2)
    assert np.array_equal(result._data, np.array([[1, 2], [3, 4]]))
    assert result.requires_grad


def test_stack_axis_1():
    """Test stacking along axis 1."""
    a = tensor.Tensor([1, 2], requires_grad=True)
    b = tensor.Tensor([3, 4], requires_grad=True)

    result = F.stack([a, b], axis=1)

    # Check the result
    assert result._data.shape == (2, 2)
    assert np.array_equal(result._data, np.array([[1, 3], [2, 4]]))
    assert result.requires_grad


def test_stack_backward():
    """Test gradient calculation for stack operation."""
    a = tensor.Tensor([1, 2], requires_grad=True)
    b = tensor.Tensor([3, 4], requires_grad=True)

    result = F.stack([a, b])

    F.sum(result).backward()

    # Check the gradients
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    assert np.array_equal(b.grad, np.array([1, 1]))


def test_stack_backward_axis_1():
    """Test gradient calculation when stacking along axis 1."""
    a = tensor.Tensor([1, 2], requires_grad=True)
    b = tensor.Tensor([3, 4], requires_grad=True)

    result = F.stack([a, b], axis=1)

    F.sum(result).backward()

    # Check the gradients
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    assert np.array_equal(b.grad, np.array([1, 1]))


def test_stack_mixed_requires_grad():
    """Test stacking tensors where only some require gradients."""
    a = tensor.Tensor([1, 2], requires_grad=True)
    b = tensor.Tensor([3, 4], requires_grad=False)

    result = F.stack([a, b])

    # Result should require gradients since at least one input requires them
    assert result.requires_grad

    F.sum(result).backward()

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    # b shouldn't have a gradient
    assert b.grad is None


def test_stack_backward_chain():
    """Test gradient flow when stack is part of a chain."""
    a = tensor.Tensor([1, 2], requires_grad=True)
    b = tensor.Tensor([3, 4], requires_grad=True)

    result1 = F.stack([a, b])
    # Stack the stacked tensor with another tensor
    c = tensor.Tensor([[5, 6], [7, 8]], requires_grad=True)
    result2 = F.stack([result1, c], axis=0)

    F.sum(result2).backward()

    # Check the gradients
    assert a.grad is not None
    assert b.grad is not None
    assert c.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    assert np.array_equal(b.grad, np.array([1, 1]))
    assert np.array_equal(c.grad, np.array([[1, 1], [1, 1]]))


def test_stack_exp_chain():
    """Test gradient flow when stack is followed by exp."""
    a = tensor.Tensor([1, 2], requires_grad=True)
    b = tensor.Tensor([3, 4], requires_grad=True)

    result1 = F.stack([a, b])
    result2 = F.exp(result1)

    F.sum(result2).backward()

    # Check the gradients - should be exp(value)
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, np.exp(a._data))
    assert np.array_equal(b.grad, np.exp(b._data))


def test_stack_sine_cosine_chain():
    """Test gradient flow when stack is part of a sin/cos chain."""
    a = tensor.Tensor([1, 2], requires_grad=True)
    b = tensor.Tensor([3, 4], requires_grad=True)

    result1 = F.stack([a, b])
    result2 = F.sin(result1)
    result3 = F.cos(result2)

    F.sum(result3).backward()

    # Check the gradients
    assert a.grad is not None
    assert b.grad is not None

    # Expected gradients: -sin(sin(value)) * cos(value)
    expected_grad_a = -np.sin(np.sin(a._data)) * np.cos(a._data)
    expected_grad_b = -np.sin(np.sin(b._data)) * np.cos(b._data)

    assert np.allclose(a.grad, expected_grad_a)
    assert np.allclose(b.grad, expected_grad_b)


def test_stack_multiple_tensors():
    """Test stacking more than two tensors."""
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    b = tensor.Tensor(np.array([3, 4]), requires_grad=True)
    c = tensor.Tensor(np.array([5, 6]), requires_grad=True)

    result = F.stack([a, b, c])

    # Check the result
    assert result._data.shape == (3, 2)
    assert np.array_equal(result._data, np.array([[1, 2], [3, 4], [5, 6]]))

    F.sum(result).backward()

    # Check the gradients
    assert a.grad is not None
    assert b.grad is not None
    assert c.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1]))
    assert np.array_equal(b.grad, np.array([1, 1]))
    assert np.array_equal(c.grad, np.array([1, 1]))


def test_stack_multidimensional_tensors():
    """Test stacking tensors with multiple dimensions."""
    a = tensor.Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
    b = tensor.Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True)

    # Stack along the first axis (creating a 3D tensor)
    result = F.stack([a, b], axis=0)

    # Check the result
    assert result._data.shape == (2, 2, 2)
    assert np.array_equal(result._data[0], a._data)
    assert np.array_equal(result._data[1], b._data)

    F.sum(result).backward()

    # Check the gradients
    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, np.ones_like(a._data))
    assert np.array_equal(b.grad, np.ones_like(b._data))
