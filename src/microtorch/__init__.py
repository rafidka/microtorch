__version__ = "0.1.0"

import numpy as np

from . import tensor

# Global RNG for reproducible weight initialization (like torch.manual_seed)
_rng: np.random.Generator = np.random.default_rng()


def manual_seed(seed: int) -> None:
    """
    Set the seed for random number generation.

    This affects weight initialization in nn.Linear, nn.Conv2d, etc.
    Similar to torch.manual_seed().

    Args:
        seed: The seed value to use.
    """
    global _rng  # noqa: PLW0603
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    """
    Get the global random number generator.

    Used internally by nn modules for weight initialization.

    Returns:
        The global numpy random Generator.
    """
    return _rng


def equal(a: tensor.Tensor, b: tensor.Tensor) -> bool:
    """
    Check if two tensors are element-wise equal.

    Args:
        a (tensor.Tensor): The first tensor to compare.
        b (tensor.Tensor): The second tensor to compare.

    Returns:
        bool: True if the tensors are element-wise equal, False otherwise.
    """
    return bool(np.equal(a._data, b._data).all())  # pyright: ignore[reportPrivateUsage]


__all__ = ["equal", "get_rng", "manual_seed", "tensor"]
