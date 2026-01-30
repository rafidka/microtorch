"""PyTorch comparison tests.

These tests verify that MicroTorch produces the same outputs as PyTorch
for equivalent operations. This provides confidence that the implementation
is correct.

All tests are skipped if PyTorch is not installed.
"""

import numpy as np
import pytest

try:
    import torch
    import torch.nn as torch_nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from microtorch.nn import CrossEntropyLoss, Linear, ReLU, Softmax
from microtorch.optim import SGD
from microtorch.tensor import Tensor, functional as F

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")

# Tolerances for comparison
RTOL = 1e-5
ATOL = 1e-6


def to_microtorch(arr: np.ndarray, requires_grad: bool = False) -> Tensor:
    """Convert numpy array to MicroTorch tensor."""
    return Tensor(arr.astype(np.float32), requires_grad=requires_grad)


def to_pytorch(arr: np.ndarray, requires_grad: bool = False) -> "torch.Tensor":
    """Convert numpy array to PyTorch tensor."""
    return torch.tensor(arr.astype(np.float32), requires_grad=requires_grad)


class TestBasicOperations:
    """Compare basic operations with PyTorch."""

    def test_add_matches_pytorch(self):
        """Test addition matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)
        b_np = np.random.randn(3, 4)

        # MicroTorch
        a_mt = to_microtorch(a_np)
        b_mt = to_microtorch(b_np)
        result_mt = (a_mt + b_mt).numpy()

        # PyTorch
        a_pt = to_pytorch(a_np)
        b_pt = to_pytorch(b_np)
        result_pt = (a_pt + b_pt).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_sub_matches_pytorch(self):
        """Test subtraction matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)
        b_np = np.random.randn(3, 4)

        result_mt = (to_microtorch(a_np) - to_microtorch(b_np)).numpy()
        result_pt = (to_pytorch(a_np) - to_pytorch(b_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_mul_matches_pytorch(self):
        """Test multiplication matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)
        b_np = np.random.randn(3, 4)

        result_mt = (to_microtorch(a_np) * to_microtorch(b_np)).numpy()
        result_pt = (to_pytorch(a_np) * to_pytorch(b_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_div_matches_pytorch(self):
        """Test division matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)
        b_np = np.random.randn(3, 4) + 2.0  # Avoid div by zero

        result_mt = (to_microtorch(a_np) / to_microtorch(b_np)).numpy()
        result_pt = (to_pytorch(a_np) / to_pytorch(b_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)


class TestMatmul:
    """Compare matrix multiplication with PyTorch."""

    def test_matmul_2d_matches_pytorch(self):
        """Test 2D matmul matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)
        b_np = np.random.randn(4, 5)

        result_mt = (to_microtorch(a_np) @ to_microtorch(b_np)).numpy()
        result_pt = (to_pytorch(a_np) @ to_pytorch(b_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_matmul_3d_batched_matches_pytorch(self):
        """Test 3D batched matmul matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(2, 3, 4)
        b_np = np.random.randn(2, 4, 5)

        result_mt = (to_microtorch(a_np) @ to_microtorch(b_np)).numpy()
        result_pt = (to_pytorch(a_np) @ to_pytorch(b_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_matmul_backward_matches_pytorch(self):
        """Test matmul gradients match PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)
        b_np = np.random.randn(4, 5)

        # MicroTorch
        a_mt = to_microtorch(a_np, requires_grad=True)
        b_mt = to_microtorch(b_np, requires_grad=True)
        c_mt = a_mt @ b_mt
        F.sum(c_mt).backward()

        # PyTorch
        a_pt = to_pytorch(a_np, requires_grad=True)
        b_pt = to_pytorch(b_np, requires_grad=True)
        c_pt = a_pt @ b_pt
        c_pt.sum().backward()

        np.testing.assert_allclose(a_mt.grad, a_pt.grad.numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(b_mt.grad, b_pt.grad.numpy(), rtol=RTOL, atol=ATOL)


class TestReductions:
    """Compare reduction operations with PyTorch."""

    def test_sum_matches_pytorch(self):
        """Test sum matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)

        result_mt = F.sum(to_microtorch(a_np)).item()
        result_pt = to_pytorch(a_np).sum().item()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_sum_axis_matches_pytorch(self):
        """Test sum with axis matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)

        result_mt = F.sum(to_microtorch(a_np), axis=0).numpy()
        result_pt = to_pytorch(a_np).sum(dim=0).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_max_matches_pytorch(self):
        """Test max matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)

        result_mt = F.max(to_microtorch(a_np)).item()
        result_pt = to_pytorch(a_np).max().item()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_max_axis_matches_pytorch(self):
        """Test max with axis matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)

        result_mt = F.max(to_microtorch(a_np), axis=0).numpy()
        result_pt = to_pytorch(a_np).max(dim=0).values.numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)


class TestElementwise:
    """Compare elementwise operations with PyTorch."""

    def test_relu_matches_pytorch(self):
        """Test ReLU matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)

        # MicroTorch
        result_mt = F.relu(to_microtorch(a_np)).numpy()

        # PyTorch
        result_pt = torch.relu(to_pytorch(a_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_relu_backward_matches_pytorch(self):
        """Test ReLU gradients match PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)

        # MicroTorch
        a_mt = to_microtorch(a_np, requires_grad=True)
        F.sum(F.relu(a_mt)).backward()

        # PyTorch
        a_pt = to_pytorch(a_np, requires_grad=True)
        torch.relu(a_pt).sum().backward()

        np.testing.assert_allclose(a_mt.grad, a_pt.grad.numpy(), rtol=RTOL, atol=ATOL)

    def test_exp_matches_pytorch(self):
        """Test exp matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4) * 0.5  # Avoid overflow

        result_mt = F.exp(to_microtorch(a_np)).numpy()
        result_pt = torch.exp(to_pytorch(a_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_sin_matches_pytorch(self):
        """Test sin matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)

        result_mt = F.sin(to_microtorch(a_np)).numpy()
        result_pt = torch.sin(to_pytorch(a_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_cos_matches_pytorch(self):
        """Test cos matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)

        result_mt = F.cos(to_microtorch(a_np)).numpy()
        result_pt = torch.cos(to_pytorch(a_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)


class TestLinearLayer:
    """Compare Linear layer with PyTorch."""

    def test_linear_forward_matches_pytorch(self):
        """Test Linear forward matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 4).astype(np.float32)
        w_np = np.random.randn(4, 3).astype(np.float32)
        b_np = np.random.randn(3).astype(np.float32)

        # MicroTorch
        linear_mt = Linear(4, 3)
        linear_mt.weight._data = w_np.copy()
        linear_mt.bias._data = b_np.copy()
        result_mt = linear_mt(Tensor(x_np)).numpy()

        # PyTorch (note: PyTorch uses transposed weights)
        linear_pt = torch_nn.Linear(4, 3)
        with torch.no_grad():
            linear_pt.weight.copy_(torch.tensor(w_np.T))
            linear_pt.bias.copy_(torch.tensor(b_np))
        result_pt = linear_pt(torch.tensor(x_np)).detach().numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_linear_backward_matches_pytorch(self):
        """Test Linear backward matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 4).astype(np.float32)
        w_np = np.random.randn(4, 3).astype(np.float32)
        b_np = np.random.randn(3).astype(np.float32)

        # MicroTorch
        linear_mt = Linear(4, 3)
        linear_mt.weight._data = w_np.copy()
        linear_mt.bias._data = b_np.copy()
        x_mt = Tensor(x_np, requires_grad=True)
        out_mt = linear_mt(x_mt)
        F.sum(out_mt).backward()

        # PyTorch
        linear_pt = torch_nn.Linear(4, 3)
        with torch.no_grad():
            linear_pt.weight.copy_(torch.tensor(w_np.T))
            linear_pt.bias.copy_(torch.tensor(b_np))
        x_pt = torch.tensor(x_np, requires_grad=True)
        out_pt = linear_pt(x_pt)
        out_pt.sum().backward()

        # Compare input gradients
        np.testing.assert_allclose(x_mt.grad, x_pt.grad.numpy(), rtol=RTOL, atol=ATOL)


class TestSoftmax:
    """Compare Softmax with PyTorch."""

    def test_softmax_forward_matches_pytorch(self):
        """Test Softmax forward matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(2, 4)

        # MicroTorch
        softmax_mt = Softmax(dim=1)
        result_mt = softmax_mt(to_microtorch(a_np)).numpy()

        # PyTorch
        softmax_pt = torch_nn.Softmax(dim=1)
        result_pt = softmax_pt(to_pytorch(a_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_softmax_backward_matches_pytorch(self):
        """Test Softmax backward matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(2, 4)

        # MicroTorch
        softmax_mt = Softmax(dim=1)
        a_mt = to_microtorch(a_np, requires_grad=True)
        F.sum(softmax_mt(a_mt)).backward()

        # PyTorch
        softmax_pt = torch_nn.Softmax(dim=1)
        a_pt = to_pytorch(a_np, requires_grad=True)
        softmax_pt(a_pt).sum().backward()

        np.testing.assert_allclose(a_mt.grad, a_pt.grad.numpy(), rtol=RTOL, atol=ATOL)


class TestCrossEntropy:
    """Compare CrossEntropyLoss with PyTorch."""

    def test_cross_entropy_forward_matches_pytorch(self):
        """Test CrossEntropyLoss forward matches PyTorch."""
        np.random.seed(42)
        logits_np = np.random.randn(4, 3)
        targets_np = np.array([0, 1, 2, 1])

        # MicroTorch
        criterion_mt = CrossEntropyLoss()
        loss_mt = criterion_mt(to_microtorch(logits_np), Tensor(targets_np)).item()

        # PyTorch
        criterion_pt = torch_nn.CrossEntropyLoss()
        loss_pt = criterion_pt(to_pytorch(logits_np), torch.tensor(targets_np)).item()

        np.testing.assert_allclose(loss_mt, loss_pt, rtol=RTOL, atol=ATOL)

    def test_cross_entropy_backward_matches_pytorch(self):
        """Test CrossEntropyLoss backward matches PyTorch."""
        np.random.seed(42)
        logits_np = np.random.randn(4, 3)
        targets_np = np.array([0, 1, 2, 1])

        # MicroTorch
        criterion_mt = CrossEntropyLoss()
        logits_mt = to_microtorch(logits_np, requires_grad=True)
        loss_mt = criterion_mt(logits_mt, Tensor(targets_np))
        loss_mt.backward()

        # PyTorch
        criterion_pt = torch_nn.CrossEntropyLoss()
        logits_pt = to_pytorch(logits_np, requires_grad=True)
        loss_pt = criterion_pt(logits_pt, torch.tensor(targets_np))
        loss_pt.backward()

        np.testing.assert_allclose(
            logits_mt.grad, logits_pt.grad.numpy(), rtol=RTOL, atol=ATOL
        )


class TestSGD:
    """Compare SGD optimizer with PyTorch."""

    def test_sgd_step_matches_pytorch(self):
        """Test SGD step matches PyTorch."""
        np.random.seed(42)
        w_np = np.random.randn(4, 3).astype(np.float32)
        grad_np = np.random.randn(4, 3).astype(np.float32)
        lr = 0.1

        # MicroTorch
        linear_mt = Linear(4, 3)
        linear_mt.weight._data = w_np.copy()
        linear_mt.weight.grad = grad_np.copy()
        optimizer_mt = SGD(linear_mt.parameters(), lr=lr)
        optimizer_mt.step()
        result_mt = linear_mt.weight.numpy()

        # PyTorch
        linear_pt = torch_nn.Linear(4, 3)
        with torch.no_grad():
            linear_pt.weight.copy_(torch.tensor(w_np.T))
        linear_pt.weight.grad = torch.tensor(grad_np.T)
        optimizer_pt = torch.optim.SGD(linear_pt.parameters(), lr=lr)
        optimizer_pt.step()
        result_pt = linear_pt.weight.detach().numpy().T

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)


class TestStack:
    """Compare stack operation with PyTorch."""

    def test_stack_forward_matches_pytorch(self):
        """Test stack forward matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)
        b_np = np.random.randn(3, 4)

        # MicroTorch
        result_mt = F.stack([to_microtorch(a_np), to_microtorch(b_np)]).numpy()

        # PyTorch
        result_pt = torch.stack([to_pytorch(a_np), to_pytorch(b_np)]).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_stack_backward_matches_pytorch(self):
        """Test stack backward matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)
        b_np = np.random.randn(3, 4)

        # MicroTorch
        a_mt = to_microtorch(a_np, requires_grad=True)
        b_mt = to_microtorch(b_np, requires_grad=True)
        F.sum(F.stack([a_mt, b_mt])).backward()

        # PyTorch
        a_pt = to_pytorch(a_np, requires_grad=True)
        b_pt = to_pytorch(b_np, requires_grad=True)
        torch.stack([a_pt, b_pt]).sum().backward()

        np.testing.assert_allclose(a_mt.grad, a_pt.grad.numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(b_mt.grad, b_pt.grad.numpy(), rtol=RTOL, atol=ATOL)


class TestBroadcasting:
    """Compare broadcasting behavior with PyTorch."""

    def test_broadcast_add_matches_pytorch(self):
        """Test broadcast add matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)
        b_np = np.random.randn(4)  # Will broadcast

        result_mt = (to_microtorch(a_np) + to_microtorch(b_np)).numpy()
        result_pt = (to_pytorch(a_np) + to_pytorch(b_np)).numpy()

        np.testing.assert_allclose(result_mt, result_pt, rtol=RTOL, atol=ATOL)

    def test_broadcast_backward_matches_pytorch(self):
        """Test broadcast backward matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3, 4)
        b_np = np.random.randn(4)

        # MicroTorch
        a_mt = to_microtorch(a_np, requires_grad=True)
        b_mt = to_microtorch(b_np, requires_grad=True)
        F.sum(a_mt + b_mt).backward()

        # PyTorch
        a_pt = to_pytorch(a_np, requires_grad=True)
        b_pt = to_pytorch(b_np, requires_grad=True)
        (a_pt + b_pt).sum().backward()

        np.testing.assert_allclose(a_mt.grad, a_pt.grad.numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(b_mt.grad, b_pt.grad.numpy(), rtol=RTOL, atol=ATOL)
