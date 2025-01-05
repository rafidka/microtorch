import numpy as np

from microtorch.nn import ReLU, Softmax
from microtorch.tensor import Tensor

# pyright: reportPrivateUsage=false


def test_relu_positive_inputs():
    relu = ReLU()
    input_tensor = Tensor([1.0, 2.0, 3.0])
    expected_output = Tensor([1.0, 2.0, 3.0])
    assert np.array_equal(relu(input_tensor)._data, expected_output._data)


def test_relu_negative_inputs():
    relu = ReLU()
    input_tensor = Tensor([-1.0, -2.0, -3.0])
    expected_output = Tensor([0.0, 0.0, 0.0])
    assert np.array_equal(relu(input_tensor)._data, expected_output._data)


def test_relu_mixed_inputs():
    relu = ReLU()
    input_tensor = Tensor([-1.0, 2.0, -3.0])
    expected_output = Tensor([0.0, 2.0, 0.0])
    assert np.array_equal(relu(input_tensor)._data, expected_output._data)


def test_relu_batched_inputs():
    relu = ReLU()
    input_tensor = Tensor([[-1.0, 2.0, -3.0], [1.0, -2.0, 3.0]])
    expected_output = Tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]])
    assert np.array_equal(relu(input_tensor)._data, expected_output._data)


def test_softmax_1d():
    softmax = Softmax()
    input_tensor = Tensor([1.0, 2.0, 3.0])
    expected_output = Tensor([0.09003057, 0.24472847, 0.66524096])
    assert np.allclose(softmax(input_tensor)._data, expected_output._data, atol=1e-7)


def test_softmax_2d_dim0():
    softmax = Softmax(dim=0)
    input_tensor = Tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    expected_output = Tensor(
        [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    assert np.allclose(softmax(input_tensor)._data, expected_output._data, atol=1e-7)


def test_softmax_2d_dim1():
    softmax = Softmax(dim=1)
    input_tensor = Tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    expected_output = Tensor(
        [
            [0.09003057, 0.24472847, 0.66524096],
            [0.09003057, 0.24472847, 0.66524096],
        ]
    )
    assert np.allclose(softmax(input_tensor)._data, expected_output._data, atol=1e-7)


def test_softmax_3d_dim0():
    softmax = Softmax(dim=0)
    input_tensor = Tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ],
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ],
        ]
    )
    expected_output = Tensor(
        [
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
        ]
    )
    assert np.allclose(softmax(input_tensor)._data, expected_output._data, atol=1e-7)


def test_softmax_3d_dim1():
    softmax = Softmax(dim=1)
    input_tensor = Tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ],
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ],
        ]
    )
    expected_output = Tensor(
        [
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
        ]
    )
    assert np.allclose(softmax(input_tensor)._data, expected_output._data, atol=1e-7)


def test_softmax_3d_dim2():
    softmax = Softmax(dim=2)
    input_tensor = Tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ],
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ],
        ]
    )
    expected_output = Tensor(
        [
            [
                [0.09003057, 0.24472847, 0.66524096],
                [0.09003057, 0.24472847, 0.66524096],
            ],
            [
                [0.09003057, 0.24472847, 0.66524096],
                [0.09003057, 0.24472847, 0.66524096],
            ],
        ]
    )
    assert np.allclose(softmax(input_tensor)._data, expected_output._data, atol=1e-7)
