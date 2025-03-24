__version__ = "0.1.0"

import numpy as np

from . import tensor


def equal(a: tensor.Tensor, b: tensor.Tensor) -> bool:
    """
    Check if two tensors are element-wise equal.

    Args:
        a (tensor.Tensor): The first tensor to compare.
        b (tensor.Tensor): The second tensor to compare.

    Returns:
        bool: True if the tensors are element-wise equal, False otherwise.
    """
    return np.equal(a._data, b._data).all()


__all__ = ["tensor", "equal"]
