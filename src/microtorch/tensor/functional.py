from typing import Any

import numpy as np

from microtorch import tensor

# pyright: reportPrivateUsage=false


def _accumulate_broadcasted_gradients(
    grad: np.ndarray[Any, Any], shape: tuple[int, ...]
):
    """
    Accumulates the gradients along the axes that were broadcasted.

    This is required during the backward phase of operations that result in broadcasting
    of tensors. In such cases, the gradients need to be summed along the axes that were
    broadcasted.

    Args:
        grad (np.ndarray): The gradients to accumulate.
        shape (tuple[int]): The shape of the tensor to accumulate the gradients for.

    Returns:
        np.ndarray: The accumulated gradients.
    """
    # Calculate the ndim of the gradient, excluding the axes at the beginning of the
    # shape that are equal to 1.
    shape_ndim = len(shape)
    for s in shape:
        if s != 1:
            break
        shape_ndim -= 1

    # Calculate the number of axes that were broadcasted.
    num_broadcasted_axes = len(grad.shape) - shape_ndim

    # Sum along the axes that were broadcasted.
    return np.sum(grad, axis=tuple(range(num_broadcasted_axes)))


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
            a.grad += _accumulate_broadcasted_gradients(out.grad, a.grad.shape)
        if b.requires_grad:
            assert b.grad is not None
            assert out.grad is not None
            b.grad += _accumulate_broadcasted_gradients(out.grad, b.grad.shape)

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
            a.grad += _accumulate_broadcasted_gradients(out.grad, a.grad.shape)
        if b.requires_grad:
            assert b.grad is not None
            assert out.grad is not None
            b.grad -= _accumulate_broadcasted_gradients(out.grad, b.grad.shape)

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
            a.grad += _accumulate_broadcasted_gradients(
                b._data * out.grad, a.grad.shape
            )
        if b.requires_grad:
            assert b.grad is not None
            b.grad += _accumulate_broadcasted_gradients(
                a._data * out.grad, b.grad.shape
            )

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
            assert out.grad is not None
            a.grad += _accumulate_broadcasted_gradients(
                (1 / b._data) * out.grad, a.grad.shape
            )
        if b.requires_grad:
            assert b.grad is not None
            assert out.grad is not None
            b.grad += _accumulate_broadcasted_gradients(
                -a._data / (b._data**2) * out.grad, b.grad.shape
            )

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


def reshape(a: "tensor.Tensor", shape: tuple[int, ...]):
    """
    Reshapes the tensor to the specified shape.

    Args:
        a (tensor.Tensor): The input tensor.
        shape (tuple[int]): The new shape.

    Returns:
        tensor.Tensor: The reshaped tensor.
    """
    out = tensor.Tensor(
        a._data.reshape(shape),
        requires_grad=a.requires_grad,
    )

    def _backward():
        if a.requires_grad:
            assert a.grad is not None
            assert out.grad is not None
            a.grad += out.grad.reshape(a._data.shape)

    out._backward = _backward
    out._prev = [a]
    out._op = "reshape"
    out._is_leaf = False
    return out


def cross_entropy(logits: "tensor.Tensor", target: "tensor.Tensor") -> "tensor.Tensor":
    """
    Computes the cross-entropy loss between the logits and the target.

    This function assumes:
      - logits is of shape (N, C), where N is the batch size and
        C is the number of classes.
      - target is of shape (N,) containing the class indices for each example
        in the range [0, C-1].

    Args:
        logits (Tensor): The input tensor containing unnormalized log probabilities.
        target (Tensor): The tensor containing the true class indices.

    Returns:
        Tensor: A scalar tensor containing the average cross-entropy loss over the batch.
    """
    # Check for shape mismatch.
    if logits.shape[0] != target.shape[0]:
        raise ValueError(
            f"Expected logits and target to have the same batch size, but got "
            f"{logits.shape[0]} and {target.shape[0]}."
        )

    # Convert logits data to NumPy arrays for computation
    logits_data = logits._data
    target_data = target._data.astype(int)

    # Number of samples in the batch
    N: int = logits_data.shape[0]

    # Compute softmax probabilities
    exps: np.ndarray[Any, Any] = np.exp(
        logits_data - np.max(logits_data, axis=1, keepdims=True)  # more stable
    )
    probs: np.ndarray[Any, Any] = exps / np.sum(exps, axis=1, keepdims=True)

    # Compute cross-entropy loss
    # clamp probabilities to avoid log(0)
    eps = 1e-15
    probs_clamped = np.clip(probs, eps, 1 - eps)
    correct_log_probs: np.ndarray[Any, Any] = -np.log(
        probs_clamped[np.arange(N), target_data]
    )
    loss_value = np.mean(correct_log_probs)

    # Create the output tensor
    out = tensor.Tensor(np.array(loss_value), requires_grad=logits.requires_grad)

    # Define backward pass
    def _backward():
        assert logits.requires_grad

        # Gradient wrt logits
        grad_logits = probs.copy()
        grad_logits[np.arange(N), target_data] -= 1.0
        grad_logits /= N  # average loss

        # Chain rule: multiply by gradient from the next node (out.grad)
        # Note that out is a scalar, so out.grad should be 1 by default,
        # but we multiply explicitly to follow the microtorch style
        assert logits.grad is not None
        assert out.grad is not None
        logits.grad += grad_logits * out.grad

    out._backward = _backward
    out._prev = [logits, target]
    out._op = "cross_entropy_loss"
    out._is_leaf = False

    return out


def stack(tensors: list["tensor.Tensor"], axis: int = 0):
    """
    Stacks a list of tensors along the specified axis.

    Args:
        tensors (list[Tensor]): The list of input tensors.
        axis (int, optional): The axis along which to stack the tensors. Default is 0.

    Returns:
        Tensor: The stacked tensor.
    """
    if not tensors:
        raise ValueError("Expected non-empty list of tensors")

    # Check that all tensors have the same shape
    first_shape = tensors[0]._data.shape
    for i, t in enumerate(tensors[1:], 1):
        if t._data.shape != first_shape:
            raise ValueError(
                f"All tensors must have the same shape. "
                f"tensor at index 0 has shape {first_shape}, "
                f"tensor at index {i} has shape {t._data.shape}"
            )

    out = tensor.Tensor(
        np.stack([t._data for t in tensors], axis=axis),
        requires_grad=np.any([t.requires_grad for t in tensors]),
    )

    def _backward():
        if not out.requires_grad:
            return

        assert out.grad is not None

        # Split the gradient along the stacked dimension and add to inputs
        grad_slices = np.split(out.grad, len(tensors), axis=axis)

        for i, t in enumerate(tensors):
            if t.requires_grad:
                assert t.grad is not None
                # Remove the stacked dimension to match original tensor shape
                t.grad += np.squeeze(grad_slices[i], axis=axis)

    out._backward = _backward
    out._prev = tensors
    out._op = "stack"
    out._is_leaf = False
    return out
