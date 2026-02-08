from collections.abc import Callable

import numpy as np


def numerical_grad_np(
    f: Callable[..., float], *inputs: np.ndarray, eps: float = 1e-5
) -> list[np.ndarray]:
    """
    Compute numerical gradients using central finite differences.

    Args:
        f: Function that takes numpy arrays and returns a scalar.
        inputs: List of numpy arrays.
        eps: Perturbation size for finite differences.

    Returns:
        List of numerical gradients, one per input array.
    """
    grads: list[np.ndarray] = []
    for inp_idx, inp in enumerate(inputs):
        if not isinstance(inp, np.ndarray):
            raise TypeError(
                f"numerical_grad_np expects NumPy inputs, got {type(inp).__name__}"
            )
        grad = np.zeros_like(inp, dtype=np.float64)

        for var_idxs in np.ndindex(inp.shape):
            inp_plus = inp.copy()
            inp_plus[var_idxs] += eps
            inp_minus = inp.copy()
            inp_minus[var_idxs] -= eps

            inputs_plus = [*inputs[:inp_idx], inp_plus, *inputs[inp_idx + 1 :]]
            inputs_minus = [*inputs[:inp_idx], inp_minus, *inputs[inp_idx + 1 :]]

            plus = f(*inputs_plus)
            minus = f(*inputs_minus)
            if not np.isscalar(plus) or not np.isscalar(minus):
                raise ValueError(
                    "numerical_grad_np expects scalar output, got "
                    f"{np.asarray(plus).shape}"
                )
            grad[var_idxs] = (plus - minus) / (2.0 * eps)
        grads.append(grad)
    return grads
