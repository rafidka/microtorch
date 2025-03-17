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


def test_tensor_move():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)
    tensor1 = x + y
    tensor2 = Tensor([])
    tensor2._move(tensor1)  # type: ignore
    assert np.array_equal(tensor1.numpy(), tensor2.numpy())
    assert tensor1.requires_grad == tensor2.requires_grad
    assert tensor1.grad is not None
    assert tensor2.grad is not None
    assert np.array_equal(tensor1.grad, tensor2.grad)


def test_tensor_add():
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 + tensor2
    assert np.array_equal(result.numpy(), np.array([5, 7, 9]))


def test_tensor_iadd():
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    tensor1 += tensor2
    assert np.array_equal(tensor1.numpy(), np.array([5, 7, 9]))


def test_tensor_sub():
    tensor1 = Tensor([4, 5, 6])
    tensor2 = Tensor([1, 2, 3])
    result = tensor1 - tensor2
    assert np.array_equal(result.numpy(), np.array([3, 3, 3]))


def test_tensor_isub():
    tensor1 = Tensor([4, 5, 6])
    tensor2 = Tensor([1, 2, 3])
    tensor1 -= tensor2
    assert np.array_equal(tensor1.numpy(), np.array([3, 3, 3]))


def test_tensor_neg():
    tensor = Tensor([1, -2, 3])
    result = -tensor
    assert np.array_equal(result.numpy(), np.array([-1, 2, -3]))


def test_tensor_mul():
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    result = tensor1 * tensor2
    assert np.array_equal(result.numpy(), np.array([4, 10, 18]))


def test_tensor_imul():
    tensor1 = Tensor([1, 2, 3])
    tensor2 = Tensor([4, 5, 6])
    tensor1 *= tensor2
    assert np.array_equal(tensor1.numpy(), np.array([4, 10, 18]))


def test_tensor_truediv():
    tensor1 = Tensor([4, 6, 8])
    tensor2 = Tensor([2, 3, 4])
    result = tensor1 / tensor2
    assert np.array_equal(result.numpy(), np.array([2, 2, 2]))


def test_tensor_itruediv():
    tensor1 = Tensor([4, 6, 8])
    tensor2 = Tensor([2, 3, 4])
    tensor1 /= tensor2
    assert np.array_equal(tensor1.numpy(), np.array([2, 2, 2]))


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
    expected_repr = (
        "Tensor(data=[1 2 3], shape=(3,), requires_grad=True, grad=[0. 0. 0.])"
    )
    assert repr(tensor) == expected_repr
