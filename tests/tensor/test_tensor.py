import numpy as np

from microtorch.tensor.tensor import Tensor


def test_tensor_initialization():
    data = [1, 2, 3]
    tensor = Tensor(data)
    assert np.array_equal(tensor.numpy(), np.array(data))
    assert tensor.requires_grad is False
    assert tensor.grad is None


def test_tensor_initialization_with_requires_grad():
    data = [1, 2, 3]
    tensor = Tensor(data, requires_grad=True)
    assert np.array_equal(tensor.numpy(), np.array(data))
    assert tensor.requires_grad is True
    assert tensor.grad is not None
    assert np.array_equal(tensor.grad, np.zeros_like(data))


def test_tensor_addition():
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 + tensor2
    assert np.array_equal(result.numpy(), np.array([5, 7, 9]))


def test_tensor_subtraction():
    tensor1 = Tensor([4, 5, 6])
    tensor2 = Tensor([1, 2, 3])
    result = tensor1 - tensor2
    assert np.array_equal(result.numpy(), np.array([3, 3, 3]))


def test_tensor_negation():
    tensor = Tensor([1, -2, 3])
    result = -tensor
    assert np.array_equal(result.numpy(), np.array([-1, 2, -3]))


def test_tensor_multiplication():
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 * tensor2
    assert np.array_equal(result.numpy(), np.array([4, 10, 18]))


def test_tensor_division():
    tensor1 = Tensor([4, 6, 8])
    tensor2 = Tensor([2, 3, 4])
    result = tensor1 / tensor2
    assert np.array_equal(result.numpy(), np.array([2, 2, 2]))


def test_tensor_matmul():
    tensor1 = Tensor([[1, 2], [3, 4]])
    tensor2 = Tensor([[5, 6], [7, 8]])
    result = tensor1 @ tensor2
    assert np.array_equal(result.numpy(), np.array([[19, 22], [43, 50]]))


def test_tensor_shape():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    assert tensor.shape == (2, 3)


def test_tensor_repr():
    tensor = Tensor([1, 2, 3], requires_grad=True)
    expected_repr = "Tensor(data=[1 2 3], requires_grad=True, grad=[0. 0. 0.])"
    assert repr(tensor) == expected_repr
