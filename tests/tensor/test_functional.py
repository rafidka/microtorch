import numpy as np

from microtorch.tensor import functional as F, tensor

# pyright: reportPrivateUsage=false


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
