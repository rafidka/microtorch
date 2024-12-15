import numpy as np

from . import tensor


def add(a: "tensor.Tensor", b: "tensor.Tensor"):
    out = tensor.Tensor(
        a.data + b.data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            a.grad += out.grad
        if b.requires_grad:
            b.grad += out.grad

    out._backward = _backward
    out._prev = [a, b]
    out._op = "add"
    out.is_leaf = False
    return out


def neg(a: "tensor.Tensor"):
    out = tensor.Tensor(
        -a.data,
        requires_grad=a.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            a.grad += -out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "neg"
    out.is_leaf = False
    return out


def mul(a: "tensor.Tensor", b: "tensor.Tensor"):
    out = tensor.Tensor(
        a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            a.grad += b.data * out.grad
        if b.requires_grad:
            b.grad += a.data * out.grad

    out._backward = _backward
    out._prev = [a, b]
    out._op = "mul"
    out.is_leaf = False
    return out


def matmul(a: "tensor.Tensor", b: "tensor.Tensor"):
    out = tensor.Tensor(
        np.matmul(a.data, b.data),
        requires_grad=a.requires_grad or b.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            a.grad += np.matmul(out.grad, b.data.T)
        if b.requires_grad:
            b.grad += np.matmul(a.data.T, out.grad)

    out._backward = _backward
    out._prev = [a, b]
    out._op = "matmul"
    out.is_leaf = False
    return out


def sin(a: "tensor.Tensor"):
    out = tensor.Tensor(np.sin(a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += np.cos(a.data) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "sin"
    out.is_leaf = False
    return out


def cos(a: "tensor.Tensor"):
    out = tensor.Tensor(np.cos(a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += -np.sin(a.data) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "sin"
    out.is_leaf = False
    return out


def exp(a: "tensor.Tensor"):
    out = tensor.Tensor(np.exp(a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += np.exp(a.data) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "exp"
    out.is_leaf = False
    return out


def sum(a: "tensor.Tensor"):
    out = tensor.Tensor(np.sum(a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "sum"
    out.is_leaf = False
    return out


def max(a: "tensor.Tensor", axis=None, keepdims=False):
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
    out.is_leaf = False
    return out


def relu(a: "tensor.Tensor"):
    out = tensor.Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad += (a.data > 0) * out.grad

    out._backward = _backward
    out._prev = [a]
    out._op = "relu"
    out.is_leaf = False
    return out
