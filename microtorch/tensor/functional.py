import numpy as np

from . import tensor

# pyright: reportPrivateUsage=false


def add(a: "tensor.Tensor", b: "tensor.Tensor"):
    """
    Adds two tensors element-wise.

    Args:
        a (tensor.Tensor): The first tensor.
        b (tensor.Tensor): The second tensor.

    Returns:
        tensor.Tensor: The result of adding the two tensors.
    """
    out = tensor.Tensor(
        a._data + b._data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            assert out.grad is not None
            a.grad += out.grad
        if b.requires_grad:
            assert b.grad is not None
            assert out.grad is not None
            b.grad += out.grad

    out._backward = _backward
    out._prev = [a, b]
    out._op = "add"
    out._is_leaf = False
    return out


def sub(a: "tensor.Tensor", b: "tensor.Tensor"):
    """
    Subtracts two tensors element-wise.

    Args:
        a (tensor.Tensor): The first tensor.
        b (tensor.Tensor): The second tensor.

    Returns:
        tensor.Tensor: The result of subtracting the second tensor from the first tensor.
    """
    out = tensor.Tensor(
        a._data - b._data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            assert out.grad is not None
            a.grad += out.grad
        if b.requires_grad:
            assert b.grad is not None
            assert out.grad is not None
            b.grad -= out.grad

    out._backward = _backward
    out._prev = [a, b]
    out._op = "sub"
    out._is_leaf = False
    return out


def neg(a: "tensor.Tensor"):
    """
    Negates the elements of the tensor.

    Args:
        a (tensor.Tensor): The input tensor.

    Returns:
        tensor.Tensor: The result of negating the input tensor.
    """
    out = tensor.Tensor(
        -a._data,
        requires_grad=a.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            assert out.grad is not None
            a.grad -= out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "neg"
    out._is_leaf = False
    return out


def mul(a: "tensor.Tensor", b: "tensor.Tensor"):
    """
    Multiplies two tensors element-wise.

    Args:
        a (tensor.Tensor): The first tensor.
        b (tensor.Tensor): The second tensor.

    Returns:
        tensor.Tensor: The result of multiplying the two tensors.
    """
    out = tensor.Tensor(
        a._data * b._data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            a.grad += b._data * out.grad
        if b.requires_grad:
            assert b.grad is not None
            b.grad += a._data * out.grad

    out._backward = _backward
    out._prev = [a, b]
    out._op = "mul"
    out._is_leaf = False
    return out


def matmul(a: "tensor.Tensor", b: "tensor.Tensor"):
    """
    Performs matrix multiplication of two tensors.

    Args:
        a (tensor.Tensor): The first tensor.
        b (tensor.Tensor): The second tensor.

    Returns:
        tensor.Tensor: The result of matrix multiplication of the two tensors.
    """
    out = tensor.Tensor(
        np.matmul(a._data, b._data),
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert out.grad is not None
            a.grad += np.matmul(out.grad, b._data.T)
        if b.requires_grad:
            assert out.grad is not None
            b.grad += np.matmul(a._data.T, out.grad)

    out._backward = _backward
    out._prev = [a, b]
    out._op = "matmul"
    out._is_leaf = False
    return out


def div(a: "tensor.Tensor", b: "tensor.Tensor"):
    """
    Divides two tensors element-wise.

    Args:
        a (tensor.Tensor): The first tensor.
        b (tensor.Tensor): The second tensor.

    Returns:
        tensor.Tensor: The result of dividing the two tensors.
    """
    out = tensor.Tensor(
        a._data / b._data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            assert b.grad is not None
            assert out.grad is not None
            a.grad += (1 / b._data) * out.grad
        if b.requires_grad:
            assert a.grad is not None
            assert b.grad is not None
            assert out.grad is not None
            b.grad += -a._data / (b._data**2) * out.grad

    out._backward = _backward
    out._prev = [a, b]
    out._op = "div"
    out._is_leaf = False
    return out


def sin(a: "tensor.Tensor"):
    """
    Computes the sine of each element in the tensor.

    Args:
        a (tensor.Tensor): The input tensor.

    Returns:
        tensor.Tensor: The result of applying the sine function element-wise.
    """
    out = tensor.Tensor(np.sin(a._data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            assert out.grad is not None
            a.grad += np.cos(a._data) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "sin"
    out._is_leaf = False
    return out


def cos(a: "tensor.Tensor"):
    """
    Computes the cosine of each element in the tensor.

    Args:
        a (tensor.Tensor): The input tensor.

    Returns:
        tensor.Tensor: The result of applying the cosine function element-wise.
    """
    out = tensor.Tensor(np.cos(a._data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            assert out.grad is not None
            a.grad += -np.sin(a._data) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "sin"
    out._is_leaf = False
    return out


def exp(a: "tensor.Tensor"):
    """
    Computes the exponential of each element in the tensor.

    Args:
        a (tensor.Tensor): The input tensor.

    Returns:
        tensor.Tensor: The result of applying the exponential function element-wise.
    """
    out = tensor.Tensor(np.exp(a._data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            assert out.grad is not None
            a.grad += np.exp(a._data) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "exp"
    out._is_leaf = False
    return out


def sum(
    a: "tensor.Tensor",
    axis: int | tuple[int] | None = None,
    keepdims: bool = False,
):
    """
    Sums the elements of the tensor along the specified axis.

    Args:
        a (tensor.Tensor): The input tensor.
        axis (int | tuple[int] | None, optional): The axis or axes along which to sum.
            Default is None.
        keepdims (bool, optional): Whether to keep the dimensions of the result.
            Default is False.

    Returns:
        tensor.Tensor: The result of summing the elements of the input tensor.
    """
    out = tensor.Tensor(
        np.sum(a._data, axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            assert out.grad is not None
            a.grad += out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "sum"
    out._is_leaf = False
    return out


def max(
    a: "tensor.Tensor",
    axis: int | tuple[int] | None = None,
    keepdims: bool = False,
):
    """
    Computes the maximum of elements along the specified axis.

    Args:
        a (tensor.Tensor): The input tensor.
        axis (int | tuple[int]): The axis or axes along which to compute the maximum.
        keepdims (bool, optional): Whether to keep the dimensions of the result.
            Default is False.

    Returns:
        tensor.Tensor: The result of computing the maximum along the specified axis.
    """
    out = tensor.Tensor(
        np.max(a._data, axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            mask = a._data == out._data
            a.grad += mask * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "max"
    out._is_leaf = False
    return out


def relu(a: "tensor.Tensor"):
    """
    Applies the ReLU (Rectified Linear Unit) function element-wise.

    Args:
        a (tensor.Tensor): The input tensor.

    Returns:
        tensor.Tensor: The result of applying the ReLU function element-wise.
    """
    out = tensor.Tensor(np.maximum(0, a._data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            assert out.grad is not None
            a.grad += (a._data > 0) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "relu"
    out._is_leaf = False
    return out
