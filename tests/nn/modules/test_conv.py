import numpy as np

from microtorch.nn import Conv2d
from microtorch.tensor import Tensor, functional as F

# pyright: reportPrivateUsage=false

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


def test_conv2d_initialization():
    conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    assert conv.weight.shape == (16, 3, 3, 3)
    assert conv.bias is not None
    assert conv.bias.shape == (16,)
    assert conv.weight.requires_grad
    assert conv.bias.requires_grad


def test_conv2d_initialization_no_bias():
    conv = Conv2d(
        in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
    )
    assert conv.weight.shape == (16, 3, 3, 3)
    assert conv.bias is None


def test_conv2d_initialization_kaiming():
    """Verify Kaiming/He initialization produces correct scale."""
    conv = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    fan_in = 32 * 3 * 3
    expected_std = np.sqrt(2.0 / fan_in)
    actual_std = conv.weight.numpy().std()
    # Allow 20% tolerance for random initialization
    assert abs(actual_std - expected_std) / expected_std < 0.2


def test_conv2d_forward():
    conv = Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    x = Tensor(_rng.standard_normal((2, 1, 28, 28)))
    output = conv(x)
    assert output.shape == (2, 8, 28, 28)


def test_conv2d_forward_stride():
    conv = Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
    x = Tensor(_rng.standard_normal((2, 1, 28, 28)))
    output = conv(x)
    assert output.shape == (2, 8, 14, 14)


def test_conv2d_forward_no_padding():
    conv = Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)
    x = Tensor(_rng.standard_normal((2, 1, 28, 28)))
    output = conv(x)
    assert output.shape == (2, 8, 26, 26)


def test_conv2d_gradient():
    conv = Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    x = Tensor(_rng.standard_normal((2, 1, 28, 28)), requires_grad=True)
    output = conv(x)
    loss = F.sum(output)
    loss.backward()

    assert conv.weight.grad is not None
    assert conv.bias is not None
    assert conv.bias.grad is not None
    assert x.grad is not None

    assert conv.weight.grad.shape == (8, 1, 3, 3)
    assert conv.bias.grad.shape == (8,)
    assert x.grad.shape == (2, 1, 28, 28)


def test_conv2d_gradient_no_bias():
    conv = Conv2d(
        in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False
    )
    x = Tensor(_rng.standard_normal((2, 1, 28, 28)), requires_grad=True)
    output = conv(x)
    loss = F.sum(output)
    loss.backward()

    assert conv.weight.grad is not None
    assert conv.bias is None
    assert x.grad is not None


def test_conv2d_batch_input():
    batch_size = 32
    conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    x = Tensor(_rng.standard_normal((batch_size, 3, 32, 32)))
    output = conv(x)
    assert output.shape == (batch_size, 16, 32, 32)


def test_conv2d_stored_attributes():
    conv = Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
    assert conv.in_channels == 3
    assert conv.out_channels == 16
    assert conv.kernel_size == 5
    assert conv.stride == 2
    assert conv.padding == 2
