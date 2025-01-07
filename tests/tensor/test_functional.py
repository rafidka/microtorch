import numpy as np

from microtorch.tensor import functional as F, tensor

# pyright: reportPrivateUsage=false


def test_add():
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


def test_sub():
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


def test_neg():
    a = tensor.Tensor(np.array([1, 2]), requires_grad=True)
    result = F.neg(a)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([-1, -2]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([-1, -1]))


def test_mul():
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


def test_matmul():
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


def test_div():
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


def test_sin():
    a = tensor.Tensor(np.array([0, np.pi / 2]), requires_grad=True)
    result = F.sin(a)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    assert np.array_equal(result._data, np.array([0, 1]))

    # Check the gradients
    assert a.grad is not None
    assert np.allclose(a.grad, np.array([1, 0]))


def test_cos():
    a = tensor.Tensor(np.array([0, np.pi / 2]), requires_grad=True)
    result = F.cos(a)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    np.testing.assert_almost_equal(result._data, np.array([1, 0]))

    # Check the gradients
    assert a.grad is not None
    assert np.allclose(a.grad, np.array([0, -1]))


def test_exp():
    a = tensor.Tensor(np.array([0, 1]), requires_grad=True)
    result = F.exp(a)

    F.sum(result).backward()  # do a sum to be able to call backward()

    # Check the result
    np.testing.assert_almost_equal(result._data, np.array([1, np.exp(1)]))

    # Check the gradients
    assert a.grad is not None
    assert np.allclose(a.grad, np.array([1, np.exp(1)]))


def test_sum():
    a = tensor.Tensor(np.array([1, 2, 3, 4]), requires_grad=True)
    result = F.sum(a)

    result.backward()

    # Check the result
    assert result._data == 10

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([1, 1, 1, 1]))


def test_max():
    a = tensor.Tensor(np.array([1, 2, 3, 4]), requires_grad=True)
    result = F.max(a)

    result.backward()

    # Check the result
    assert result._data == 4

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([0, 0, 0, 1]))


def test_relu():
    a = tensor.Tensor(np.array([-1, 2, 3, -4]), requires_grad=True)
    result = F.relu(a)

    F.sum(result).backward()

    # Check the result
    assert np.array_equal(result._data, np.array([0, 2, 3, 0]))

    # Check the gradients
    assert a.grad is not None
    assert np.array_equal(a.grad, np.array([0, 1, 1, 0]))


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
