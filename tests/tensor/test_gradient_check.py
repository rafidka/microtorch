"""Numerical gradient verification tests.

These tests compare analytical gradients computed by autograd with numerical
gradients computed via finite differences. This is the gold standard for
validating gradient correctness in an autograd system.
"""

import numpy as np
import pytest

from microtorch.nn import Linear, ReLU, Softmax
from microtorch.tensor import Tensor, functional as F


def numerical_gradient(
    func,
    inputs: list[Tensor],
    eps: float = 1e-5,
) -> list[np.ndarray]:
    """Compute numerical gradients using central finite differences.

    Args:
        func: Function that takes tensors and returns a scalar tensor.
        inputs: List of input tensors.
        eps: Perturbation size for finite differences.

    Returns:
        List of numerical gradients, one per input tensor.
    """
    numerical_grads = []

    for input_idx, inp in enumerate(inputs):
        grad = np.zeros_like(inp.numpy())
        flat_data = inp.numpy().flatten()

        for i in range(len(flat_data)):
            # Perturb positively
            flat_plus = flat_data.copy()
            flat_plus[i] += eps
            inputs_plus = [
                Tensor(flat_plus.reshape(inp.shape)) if j == input_idx else t
                for j, t in enumerate(inputs)
            ]
            out_plus = func(*inputs_plus)

            # Perturb negatively
            flat_minus = flat_data.copy()
            flat_minus[i] -= eps
            inputs_minus = [
                Tensor(flat_minus.reshape(inp.shape)) if j == input_idx else t
                for j, t in enumerate(inputs)
            ]
            out_minus = func(*inputs_minus)

            # Central difference
            grad.flat[i] = (out_plus.item() - out_minus.item()) / (2 * eps)

        numerical_grads.append(grad)

    return numerical_grads


def check_gradients(
    func,
    inputs: list[Tensor],
    eps: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> None:
    """Check that analytical gradients match numerical gradients.

    Args:
        func: Function that takes tensors and returns a scalar tensor.
        inputs: List of input tensors with requires_grad=True.
        eps: Perturbation size for finite differences.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Raises:
        AssertionError: If gradients don't match within tolerance.
    """
    # Compute analytical gradients
    for inp in inputs:
        if inp.grad is not None:
            inp.grad.fill(0)  # Reset gradients

    output = func(*inputs)
    output.backward()

    analytical_grads = [inp.grad for inp in inputs]

    # Compute numerical gradients
    num_grads = numerical_gradient(func, inputs, eps)

    # Compare
    for i, (analytical, numerical) in enumerate(zip(analytical_grads, num_grads)):
        assert analytical is not None, f"Input {i} has no gradient"
        if not np.allclose(analytical, numerical, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(analytical - numerical))
            rel_diff = np.max(
                np.abs(analytical - numerical) / (np.abs(numerical) + 1e-8)
            )
            raise AssertionError(
                f"Gradient mismatch for input {i}:\n"
                f"  Max absolute diff: {max_diff}\n"
                f"  Max relative diff: {rel_diff}\n"
                f"  Analytical:\n{analytical}\n"
                f"  Numerical:\n{numerical}"
            )


class TestGradientCheckBasicOps:
    """Gradient checks for basic arithmetic operations."""

    def test_grad_check_add(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.add(x, y)), [a, b])

    def test_grad_check_add_broadcast(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.add(x, y)), [a, b])

    def test_grad_check_sub(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.sub(x, y)), [a, b])

    def test_grad_check_sub_broadcast(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(1, 4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.sub(x, y)), [a, b])

    def test_grad_check_mul(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.mul(x, y)), [a, b])

    def test_grad_check_mul_broadcast(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.mul(x, y)), [a, b])

    def test_grad_check_div(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        # Avoid division by zero
        b = Tensor(np.random.randn(3, 4) + 2.0, requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.div(x, y)), [a, b])

    def test_grad_check_div_broadcast(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4) + 2.0, requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.div(x, y)), [a, b])

    def test_grad_check_neg(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.neg(x)), [a])


class TestGradientCheckMatmul:
    """Gradient checks for matrix multiplication."""

    def test_grad_check_matmul_2d(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 5), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.matmul(x, y)), [a, b])

    def test_grad_check_matmul_2d_square(self):
        a = Tensor(np.random.randn(4, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.matmul(x, y)), [a, b])

    def test_grad_check_matmul_3d_batched(self):
        """Test batched matmul (3D tensors) - this uses the new swapaxes code."""
        a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        b = Tensor(np.random.randn(2, 4, 5), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.matmul(x, y)), [a, b])

    def test_grad_check_matmul_vector(self):
        """Test matrix-vector multiplication."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.matmul(x, y)), [a, b])

    def test_grad_check_vector_matmul(self):
        """Test vector-matrix multiplication (1D @ 2D)."""
        a = Tensor(np.random.randn(3), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.matmul(x, y)), [a, b])


class TestGradientCheckReductions:
    """Gradient checks for reduction operations."""

    def test_grad_check_sum_no_axis(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(x), [a])

    def test_grad_check_sum_axis_0(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.sum(x, axis=0)), [a])

    def test_grad_check_sum_axis_1(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.sum(x, axis=1)), [a])

    def test_grad_check_sum_axis_tuple(self):
        a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.sum(x, axis=(0, 2))), [a])

    def test_grad_check_sum_keepdims(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.sum(x, axis=0, keepdims=True)), [a])

    def test_grad_check_max_no_axis(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.max(x), [a])

    def test_grad_check_max_axis_0(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.max(x, axis=0)), [a])

    def test_grad_check_max_axis_1(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.max(x, axis=1)), [a])

    def test_grad_check_max_keepdims(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.max(x, axis=0, keepdims=True)), [a])


class TestGradientCheckElementwise:
    """Gradient checks for elementwise operations."""

    def test_grad_check_sin(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.sin(x)), [a])

    def test_grad_check_cos(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.cos(x)), [a])

    def test_grad_check_exp(self):
        # Use smaller values to avoid overflow
        a = Tensor(np.random.randn(3, 4) * 0.5, requires_grad=True)
        check_gradients(lambda x: F.sum(F.exp(x)), [a])

    def test_grad_check_relu(self):
        # ReLU is non-smooth at 0, so avoid values close to 0
        data = np.random.randn(3, 4)
        data[np.abs(data) < 0.1] = 0.5  # Push away from 0
        a = Tensor(data, requires_grad=True)
        check_gradients(lambda x: F.sum(F.relu(x)), [a])


class TestGradientCheckShapeOps:
    """Gradient checks for shape operations."""

    def test_grad_check_reshape(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.reshape(x, (2, 6))), [a])

    def test_grad_check_reshape_flatten(self):
        a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        check_gradients(lambda x: F.sum(F.reshape(x, (24,))), [a])

    def test_grad_check_stack_axis_0(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.stack([x, y], axis=0)), [a, b])

    def test_grad_check_stack_axis_1(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        check_gradients(lambda x, y: F.sum(F.stack([x, y], axis=1)), [a, b])


class TestGradientCheckLosses:
    """Gradient checks for loss functions."""

    def test_grad_check_cross_entropy(self):
        # Logits
        logits = Tensor(np.random.randn(4, 3), requires_grad=True)
        # Targets (class indices)
        targets = Tensor(np.array([0, 1, 2, 1]))

        def ce_loss(x):
            return F.cross_entropy(x, targets)

        check_gradients(ce_loss, [logits])

    def test_grad_check_cross_entropy_single(self):
        logits = Tensor(np.random.randn(1, 5), requires_grad=True)
        targets = Tensor(np.array([3]))

        def ce_loss(x):
            return F.cross_entropy(x, targets)

        check_gradients(ce_loss, [logits])


class TestGradientCheckModules:
    """Gradient checks for neural network modules."""

    def test_grad_check_linear(self):
        np.random.seed(42)
        linear = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4), requires_grad=True)

        def forward(inp):
            return F.sum(linear(inp))

        check_gradients(forward, [x])

    def test_grad_check_softmax(self):
        softmax = Softmax(dim=1)
        x = Tensor(np.random.randn(2, 4), requires_grad=True)

        def forward(inp):
            return F.sum(softmax(inp))

        check_gradients(forward, [x])

    def test_grad_check_relu_module(self):
        relu = ReLU()
        # Avoid values close to 0
        data = np.random.randn(2, 4)
        data[np.abs(data) < 0.1] = 0.5
        x = Tensor(data, requires_grad=True)

        def forward(inp):
            return F.sum(relu(inp))

        check_gradients(forward, [x])


class TestGradientCheckChained:
    """Gradient checks for chained operations."""

    def test_grad_check_linear_relu_chain(self):
        np.random.seed(42)
        linear = Linear(4, 3)
        relu = ReLU()
        x = Tensor(np.random.randn(2, 4), requires_grad=True)

        def forward(inp):
            return F.sum(relu(linear(inp)))

        check_gradients(forward, [x])

    def test_grad_check_mlp_forward(self):
        """Test a simple 2-layer MLP."""
        np.random.seed(42)
        linear1 = Linear(4, 8)
        linear2 = Linear(8, 3)
        relu = ReLU()
        x = Tensor(np.random.randn(2, 4), requires_grad=True)

        def forward(inp):
            h = relu(linear1(inp))
            return F.sum(linear2(h))

        check_gradients(forward, [x])

    def test_grad_check_complex_expression(self):
        """Test a complex expression with multiple operations."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)

        def forward(x, y):
            z = F.mul(x, y) + F.sin(x)
            z = F.exp(z * Tensor(np.array(0.1)))  # Scale to avoid overflow
            return F.sum(z)

        check_gradients(forward, [a, b])
