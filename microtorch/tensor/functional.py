from typing import Union
import numpy as np

from . import tensor


# pyright: reportPrivateUsage=false


def add(a: "tensor.Tensor", b: "tensor.Tensor"):
    out = tensor.Tensor(
        a.data + b.data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad
            assert out.grad
            a.grad += out.grad
        if b.requires_grad:
            assert b.grad
            assert out.grad
            b.grad += out.grad

    out._backward = _backward
    out._prev = [a, b]
    out._op = "add"
    out._is_leaf = False
    return out


def neg(a: "tensor.Tensor"):
    out = tensor.Tensor(
        -a.data,
        requires_grad=a.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad
            assert out.grad
            a.grad -= out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "neg"
    out._is_leaf = False
    return out


def mul(a: "tensor.Tensor", b: "tensor.Tensor"):
    out = tensor.Tensor(
        a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad
            a.grad += b.data * out.grad
        if b.requires_grad:
            assert b.grad
            b.grad += a.data * out.grad

    out._backward = _backward
    out._prev = [a, b]
    out._op = "mul"
    out._is_leaf = False
    return out


def matmul(a: "tensor.Tensor", b: "tensor.Tensor"):
    out = tensor.Tensor(
        np.matmul(a.data, b.data),
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert out.grad
            a.grad += np.matmul(out.grad, b.data.T)
        if b.requires_grad:
            assert out.grad
            b.grad += np.matmul(a.data.T, out.grad)

    out._backward = _backward
    out._prev = [a, b]
    out._op = "matmul"
    out._is_leaf = False
    return out


def div(a: "tensor.Tensor", b: "tensor.Tensor"):
    out = tensor.Tensor(
        a.data / b.data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad
            assert b.grad
            assert out.grad
            a.grad += (1 / b.data) * out.grad
        if b.requires_grad:
            assert a.grad
            assert b.grad
            assert out.grad
            b.grad += -a.data / (b.data**2) * out.grad

    out._backward = _backward
    out._prev = [a, b]
    out._op = "div"
    out._is_leaf = False
    return out


def sin(a: "tensor.Tensor"):
    out = tensor.Tensor(np.sin(a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            assert a.grad
            assert out.grad
            a.grad += np.cos(a.data) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "sin"
    out._is_leaf = False
    return out


def cos(a: "tensor.Tensor"):
    out = tensor.Tensor(np.cos(a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            assert a.grad
            assert out.grad
            a.grad += -np.sin(a.data) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "sin"
    out._is_leaf = False
    return out


def exp(a: "tensor.Tensor"):
    out = tensor.Tensor(np.exp(a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            assert a.grad
            assert out.grad
            a.grad += np.exp(a.data) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "exp"
    out._is_leaf = False
    return out


def sum(a: "tensor.Tensor", axis: Union[int, tuple[int]], keepdims: bool = False):
    out = tensor.Tensor(
        np.sum(a.data, axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad
            assert out.grad
            a.grad += out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "sum"
    out._is_leaf = False
    return out


def max(a: "tensor.Tensor", axis: Union[int, tuple[int]], keepdims: bool = False):
    out = tensor.Tensor(
        np.max(a.data, axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            mask = a.data == out.data
            a.grad += mask * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "max"
    out._is_leaf = False
    return out


def relu(a: "tensor.Tensor"):
    out = tensor.Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            assert a.grad
            assert out.grad
            a.grad += (a.data > 0) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "relu"
    out._is_leaf = False
    return out
