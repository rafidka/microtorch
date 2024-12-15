# Python imports
from collections import deque

# 3rd-party imports
import numpy as np

# Local imports
from . import tensor


def _create_torder_map(tensor: "tensor.Tensor") -> dict["tensor.Tensor", int]:
    """
    Create a topological order map for the computational graph of a given tensor.

    Parameters:
    tensor (Tensor): The input tensor.

    Returns:
    dict[Tensor, int]: A dictionary mapping each tensor to its topological order.

    """
    queue = deque([tensor])
    torder_map: dict["tensor.Tensor", int] = {tensor: 1}

    while queue:
        cursor = queue.popleft()
        cursor_torder = torder_map[cursor]

        # Update the topological order of the previous tensors
        for prev in cursor._prev:  # type: ignore
            prev_torder = torder_map.get(prev, 1)
            prev_torder = max(prev_torder, cursor_torder + 1)
            torder_map[prev] = prev_torder
            queue.append(prev)

    return torder_map


def backward(tensor: "tensor.Tensor") -> None:
    """
    Performs backpropagation on a scalar tensor.

    Args:
        tensor (Tensor): The tensor to perform backpropagation on.

    Raises:
        Exception: If the tensor does not have requires_grad enabled.
        Exception: If the tensor is not a scalar tensor.

    Notes:
        - This function assigns a topological order to each tensor in the computation
        graph.
        - The gradient of the tensor is set to an array of ones with the same shape as
        the tensor's data.
        - The backward pass is performed by executing the _backward method of each
        tensor in reverse topological order.
    """
    if not tensor.requires_grad:
        raise Exception("This tensor does not requires_grad enabled.")
    if tensor.data.size != 1:
        raise Exception("Can only do backward on scalar tensors.")

    tensor.grad = np.ones(tensor.data.shape)

    torder_map = _create_torder_map(tensor)

    for tensor in map(
        lambda x: x[0],  # Get the tensor from the tuple
        sorted(
            torder_map.items(),
            key=lambda x: x[1],  # Sort by topological order
        ),
    ):
        if tensor._backward:  # type: ignore
            tensor._backward()  # type: ignore
