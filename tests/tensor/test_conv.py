import numpy as np
import pytest
import torch
from numpy.random import default_rng

from microtorch.tensor import Tensor, functional as F
from tests.utils.grad import numerical_grad_np

rnd = default_rng()


# ===============================================================================
# _im2col_np tests
# ===============================================================================


@pytest.mark.parametrize(
    ("im", "kernel", "stride", "pad", "expected_output_shape"),
    [
        (rnd.random((1, 1, 4, 4)), 2, 1, 0, (1, 4, 9)),
        (rnd.random((2, 3, 5, 5)), 3, 1, 1, (2, 27, 25)),
        (rnd.random((1, 1, 6, 6)), 3, 2, 0, (1, 9, 4)),
        (rnd.random((1, 2, 5, 5)), 3, 1, 0, (1, 18, 9)),
    ],
)
def test_im2col_np_output_shape(
    im: np.ndarray,
    kernel: int,
    stride: int,
    pad: int,
    expected_output_shape: tuple[int, ...],
):
    out = F._im2col_np(im, kernel, stride, pad)
    assert out.shape == expected_output_shape


def test_im2col_np_output_values():
    im = np.array(
        [
            [
                [
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                ],
            ],
        ]
    )  # shape (1, 1, 3, 3)
    output = F._im2col_np(im, 2, 1, 0)

    expected_output = np.array(
        [
            [
                [0, 1, 3, 4],
                [1, 2, 4, 5],
                [3, 4, 6, 7],
                [4, 5, 7, 8],
            ]
        ]
    )  # shape (1, 4, 4)

    np.testing.assert_array_equal(output, expected_output)


def test_im2col_np_output_values_multiple_channels():
    im = np.array(
        [
            [
                [
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                ],  # channel 0
                [
                    [9, 10, 11],
                    [12, 13, 14],
                    [15, 16, 17],
                ],  # channel 1
            ],
        ]
    )
    output = F._im2col_np(im, 2, 1, 0)

    expected_output = np.array(
        [
            [
                [0, 1, 3, 4],
                [1, 2, 4, 5],
                [3, 4, 6, 7],
                [4, 5, 7, 8],
                [9, 10, 12, 13],
                [10, 11, 13, 14],
                [12, 13, 15, 16],
                [13, 14, 16, 17],
            ]
        ]
    )  # shape (1, 4, 4)
    np.testing.assert_array_equal(output, expected_output)


def test_im2col_np_padding():
    im = np.array(
        [  # N
            [  # C
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ]
    )

    output = F._im2col_np(im, 2, 1, 1)

    # Image matrix after padding becomes:
    # 0, 0, 0, 0
    # 0, 1, 2, 0
    # 0, 3, 4, 0
    # 0, 0, 0, 0

    expected_output = np.array(
        [
            [
                [0, 0, 0, 0, 1, 2, 0, 3, 4],
                [0, 0, 0, 1, 2, 0, 3, 4, 0],
                [0, 1, 2, 0, 3, 4, 0, 0, 0],
                [1, 2, 0, 3, 4, 0, 0, 0, 0],
            ]
        ]
    )
    np.testing.assert_array_equal(output, expected_output)


def test_im2col_np_stride():
    im = np.array(
        [  # N
            [  # C
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ]
    )

    output = F._im2col_np(im, 2, 2, 0)

    expected_output = np.array(
        [
            [
                [1, 3, 9, 11],
                [2, 4, 10, 12],
                [5, 7, 13, 15],
                [6, 8, 14, 16],
            ]
        ]
    )
    np.testing.assert_array_equal(output, expected_output)


# ===============================================================================
# _col2im_np tests
# ===============================================================================


@pytest.mark.parametrize(
    ("cols", "kernel", "stride", "pad", "expected_im_shape"),
    [
        (rnd.random((1, 4, 9)), 2, 1, 0, (1, 1, 4, 4)),
        (rnd.random((2, 27, 25)), 3, 1, 1, (2, 3, 5, 5)),
        (rnd.random((1, 9, 4)), 3, 2, 0, (1, 1, 6, 6)),
        (rnd.random((1, 18, 9)), 3, 1, 0, (1, 2, 5, 5)),
    ],
)
def test_col2im_np_output_shape(
    cols: np.ndarray,
    kernel: int,
    stride: int,
    pad: int,
    expected_im_shape: tuple[int, ...],
):
    # `cols` here is intentionally random and does NOT represent a valid im2col
    # output for the given kernel/stride/pad configuration. A real col2im input
    # has strict structural constraints coming from sliding-window extraction
    # (column count = out_h * out_w, consistent overlaps, repeated pixels, etc.).
    #
    # This test is therefore shape-only: it verifies that `_col2im_np` reconstructs
    # an array with the correct `(N, C, H, W)` geometry when given a compatible
    # shape hint, not that the numerical values are meaningful.
    im = F._col2im_np(cols, expected_im_shape, kernel, stride, pad)
    assert im.shape == expected_im_shape


def test_col2im_np_round_trip_no_overlap():
    # When stride = kernel, there is no overlap between the windows, so the
    # col2im reconstruction should be exact.
    im = rnd.random((1, 1, 4, 4))
    cols = F._im2col_np(im, 2, 2, 0)
    im_reconstructed = F._col2im_np(cols, im.shape, 2, 2, 0)
    np.testing.assert_array_equal(im, im_reconstructed)


def test_col2im_np_round_trip_with_overlap():
    cols = np.array(
        [
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        ]
    )
    im_reconstructed = F._col2im_np(cols, (1, 1, 3, 3), 2, 1, 0)

    np.testing.assert_array_equal(
        im_reconstructed,
        np.array(
            [
                [
                    [
                        [1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1],
                    ]
                ]
            ]
        ),
    )


@pytest.mark.parametrize(("pad"), list(range(1, 5)))
def test_col2im_np_padding_is_removed(pad: int):
    im = np.array(
        [
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ]
        ]
    )
    cols = F._im2col_np(im, 2, 1, pad)
    im_reconstructed = F._col2im_np(cols, im.shape, 2, 1, pad)

    assert im_reconstructed.shape == im.shape


def test_col2im_np_multiple_channels():
    im = np.array(
        [
            [
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                [
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                ],
            ]
        ]
    )
    cols = F._im2col_np(im, 2, 2, 0)
    assert cols.shape == (1, 8, 4)

    im_reconstructed = F._col2im_np(cols, im.shape, 2, 2, 0)

    # Assert that channel 0 is all 1s and channel 1 is all 2s
    assert np.all(im_reconstructed[0, 0, :, :] == 1)
    assert np.all(im_reconstructed[0, 1, :, :] == 2)


# ===============================================================================
# im2col (Tensor version) tests
# ===============================================================================


def test_im2col_tensor_matches_numpy():
    """Tensor im2col should produce the same output as numpy _im2col_np."""

    input_np = rnd.standard_normal((2, 3, 5, 5))
    input_tensor = Tensor(input_np)

    result_tensor = F.im2col(input_tensor, kernel=3, stride=1, padding=1)
    result_numpy = F._im2col_np(input_np, kernel=3, stride=1, padding=1)

    np.testing.assert_array_equal(result_tensor.numpy(), result_numpy)


def test_im2col_tensor_requires_grad_true():
    """Output should have requires_grad=True when input does."""

    x = Tensor(np.ones((1, 1, 3, 3)), requires_grad=True)
    out = F.im2col(x, kernel=2, stride=1, padding=0)
    assert out.requires_grad is True


def test_im2col_tensor_requires_grad_false():
    """Output should have requires_grad=False when input does not."""

    x = Tensor(np.ones((1, 1, 3, 3)), requires_grad=False)
    out = F.im2col(x, kernel=2, stride=1, padding=0)
    assert out.requires_grad is False


def test_im2col_backward_shape():
    """Gradient should have the same shape as input after backward."""

    x = Tensor(rnd.standard_normal((2, 3, 5, 5)), requires_grad=True)
    out = F.im2col(x, kernel=3, stride=1, padding=1)
    loss = F.sum(out)
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_im2col_gradient_numerical():
    """Verify backward pass is mathematically correct using finite differences."""

    x_data = rnd.standard_normal((1, 2, 4, 4))

    def f(x_np: np.ndarray) -> float:
        x = Tensor(x_np, requires_grad=True)
        out = F.im2col(x, kernel=2, stride=1, padding=0)
        return float(F.sum(out).numpy())

    # Compute analytical gradient
    x = Tensor(x_data.copy(), requires_grad=True)
    out = F.im2col(x, kernel=2, stride=1, padding=0)
    loss = F.sum(out)
    loss.backward()
    analytical_grad = x.grad.copy()

    # Compute numerical gradient
    eps = 1e-5
    numerical_grad = np.zeros_like(x_data)
    for idx in np.ndindex(x_data.shape):
        x_plus = x_data.copy()
        x_plus[idx] += eps
        x_minus = x_data.copy()
        x_minus[idx] -= eps
        numerical_grad[idx] = (f(x_plus) - f(x_minus)) / (2 * eps)

    np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6)


def test_im2col_gradient_accumulation():
    """Verify gradients accumulate correctly for overlapping patches."""

    # 3x3 input, 2x2 kernel, stride=1 → patches overlap
    x = Tensor(np.ones((1, 1, 3, 3)), requires_grad=True)
    out = F.im2col(x, kernel=2, stride=1, padding=0)

    # Backward with all-ones gradient
    loss = F.sum(out)
    loss.backward()

    # Center element (1,1) appears in 4 patches → gradient should be 4
    # Corner elements appear in 1 patch → gradient should be 1
    # Edge elements appear in 2 patches → gradient should be 2
    expected = np.array([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype=np.float64)
    np.testing.assert_array_equal(x.grad, expected)


# ===============================================================================
# conv2d tests
# ===============================================================================


@pytest.mark.parametrize(
    ("input_shape", "weight_shape", "stride", "padding", "expected_output_shape"),
    [
        ((1, 1, 5, 5), (1, 1, 3, 3), 1, 0, (1, 1, 3, 3)),
        ((2, 3, 8, 8), (16, 3, 3, 3), 1, 1, (2, 16, 8, 8)),
        ((1, 1, 6, 6), (4, 1, 3, 3), 2, 0, (1, 4, 2, 2)),
    ],
)
def test_conv2d_output_shape(
    input_shape, weight_shape, stride, padding, expected_output_shape
):
    input = Tensor(rnd.random(input_shape))
    weight = Tensor(rnd.random(weight_shape))
    output = F.conv2d(input, weight, stride=stride, padding=padding)
    assert output.shape == expected_output_shape


@pytest.mark.parametrize(
    ("input_shape", "weight_shape", "stride", "padding"),
    [
        ((1, 1, 5, 5), (1, 1, 3, 3), 1, 0),
        ((2, 3, 8, 8), (16, 3, 3, 3), 1, 1),
        ((1, 1, 6, 6), (4, 1, 3, 3), 2, 0),
    ],
)
def test_conv2d_forward_matches_pytorch(input_shape, weight_shape, stride, padding):

    input_np = rnd.random(input_shape)
    weight_np = rnd.random(weight_shape)
    input = Tensor(input_np)
    weight = Tensor(weight_np)
    output = F.conv2d(input, weight, stride=stride, padding=padding)
    output_pytorch = torch.nn.functional.conv2d(
        torch.tensor(input_np),
        torch.tensor(weight_np),
        stride=stride,
        padding=padding,
    )
    np.testing.assert_allclose(output.numpy(), output_pytorch.numpy())


@pytest.mark.parametrize(
    ("input_shape", "weight_shape", "stride", "padding"),
    [
        ((1, 1, 5, 5), (1, 1, 3, 3), 1, 0),
        ((2, 3, 8, 8), (16, 3, 3, 3), 1, 1),
        ((1, 1, 6, 6), (4, 1, 3, 3), 2, 0),
    ],
)
def test_conv2d_gradient_numerical(input_shape, weight_shape, stride, padding):
    input = Tensor(rnd.random(input_shape), requires_grad=True)
    weight = Tensor(rnd.random(weight_shape), requires_grad=True)
    bias = Tensor(rnd.random((weight_shape[0],)), requires_grad=True)

    def f(input_np: np.ndarray, weight_np: np.ndarray, bias_np: np.ndarray):
        input = Tensor(input_np, requires_grad=False)
        weight = Tensor(weight_np, requires_grad=False)
        bias = Tensor(bias_np, requires_grad=False)
        return (
            F.sum(F.conv2d(input, weight, bias, stride=stride, padding=padding))
            .numpy()
            .item()
        )

    input_num_grad, weight_num_grad, bias_num_grad = numerical_grad_np(
        f, input.numpy(), weight.numpy(), bias.numpy()
    )

    out = F.sum(F.conv2d(input, weight, bias, stride=stride, padding=padding))
    out.backward()

    np.testing.assert_allclose(input.grad, input_num_grad, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(weight.grad, weight_num_grad, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(bias.grad, bias_num_grad, rtol=1e-4, atol=1e-6)


def test_conv2d_bias_addition():
    # Use a simple case where we can verify the bias manually:
    # - Input: all zeros
    # - Weight: all zeros
    # - Bias: [1, 2, 3] for 3 output channels
    #
    # With zero input and weight, output should equal bias broadcast
    # over spatial dimensions.

    input = Tensor(np.zeros((1, 1, 3, 3)))  # (N=1, C_in=1, H=3, W=3)
    weight = Tensor(np.zeros((3, 1, 2, 2)))  # (C_out=3, C_in=1, k=2, k=2)
    bias = Tensor(np.array([1.0, 2.0, 3.0]))

    output = F.conv2d(input, weight, bias, stride=1, padding=0)
    # Output shape: (1, 3, 2, 2)

    # Each output channel should be filled with its corresponding bias value
    assert output.shape == (1, 3, 2, 2)
    assert np.all(output.numpy()[0, 0, :, :] == 1.0)  # channel 0 = bias[0]
    assert np.all(output.numpy()[0, 1, :, :] == 2.0)  # channel 1 = bias[1]
    assert np.all(output.numpy()[0, 2, :, :] == 3.0)  # channel 2 = bias[2]


def test_conv2d_invalid_input_dim():
    # Input must be 4D
    input = Tensor(np.zeros((3, 3)))  # 2D - wrong
    weight = Tensor(np.zeros((1, 1, 2, 2)))
    with pytest.raises(ValueError, match="4D"):
        F.conv2d(input, weight)


def test_conv2d_invalid_weight_dim():
    # Weight must be 4D
    input = Tensor(np.zeros((1, 1, 3, 3)))
    weight = Tensor(np.zeros((1, 2, 2)))  # 3D - wrong
    with pytest.raises(ValueError, match="4D"):
        F.conv2d(input, weight)


def test_conv2d_invalid_bias_dim():
    # Bias must be 1D
    input = Tensor(np.zeros((1, 1, 3, 3)))
    weight = Tensor(np.zeros((1, 1, 2, 2)))
    bias = Tensor(np.zeros((1, 1)))  # 2D - wrong
    with pytest.raises(ValueError, match="1D"):
        F.conv2d(input, weight, bias)


def test_conv2d_channel_mismatch():
    # Input channels must match weight's C_in
    input = Tensor(np.zeros((1, 3, 5, 5)))  # C_in = 3
    weight = Tensor(np.zeros((1, 2, 2, 2)))  # C_in = 2 - mismatch!
    with pytest.raises(ValueError, match="C dimension"):
        F.conv2d(input, weight)


def test_conv2d_invalid_bias_shape():
    # Bias shape must match C_out
    input = Tensor(np.zeros((1, 1, 5, 5)))
    weight = Tensor(np.zeros((4, 1, 3, 3)))  # C_out = 4
    bias = Tensor(np.zeros((2,)))  # Wrong: should be (4,)
    with pytest.raises(ValueError, match="bias tensor to have shape"):
        F.conv2d(input, weight, bias)


def test_conv2d_non_square_kernel():
    # Kernel height and width must be the same
    input = Tensor(np.zeros((1, 1, 5, 5)))
    weight = Tensor(np.zeros((1, 1, 3, 2)))  # kh=3, kw=2 - not square!
    with pytest.raises(ValueError, match="kernel height and width"):
        F.conv2d(input, weight)


def test_im2col_np_invalid_input_dim():
    # _im2col_np requires 4D input
    invalid_input = np.zeros((3, 3, 3))  # 3D - wrong
    with pytest.raises(ValueError, match="4D"):
        F._im2col_np(invalid_input, kernel=2, stride=1, padding=0)


def test_im2col_backward_no_grad():
    # Test im2col backward when input doesn't require grad
    # This tests the branch where input.requires_grad is False (line 786->exit)
    x = Tensor(rnd.standard_normal((1, 1, 4, 4)), requires_grad=False)
    out = F.im2col(x, kernel=2, stride=1, padding=0)

    # out shape is (1, 4, 9), out.requires_grad should be False
    assert out.requires_grad is False

    # Multiply by a tensor that requires grad to create a computation graph
    multiplier = Tensor(np.ones_like(out.numpy()), requires_grad=True)
    result = F.mul(out, multiplier)

    # Sum and backward
    loss = F.sum(result)
    loss.backward()

    # x should have no gradient since requires_grad=False
    assert x.grad is None
    # multiplier should have gradient
    assert multiplier.grad is not None


def test_col2im_np_invalid_cols_shape():
    # _col2im_np requires cols with correct shape
    input_shape = (1, 1, 4, 4)  # N=1, C=1, H=4, W=4
    kernel, stride, padding = 2, 1, 0
    # H_out = (4 + 0 - 2) // 1 + 1 = 3, W_out = 3
    # Expected cols shape: (1, 1*2*2, 3*3) = (1, 4, 9)
    wrong_cols = np.zeros((1, 4, 16))  # Wrong: should be (1, 4, 9)
    with pytest.raises(ValueError, match="Expected columns to have shape"):
        F._col2im_np(wrong_cols, input_shape, kernel, stride, padding)
