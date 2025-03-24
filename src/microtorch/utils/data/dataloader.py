import random
from collections.abc import Iterator
from typing import Any

from microtorch.tensor import Tensor, functional as F

from .dataset import Dataset


def default_collate_fn(batch: list[tuple[Any, ...]]) -> tuple[list[Any] | Tensor, ...]:
    """
    Default collation function for batching data.

    This function takes a list of tuples and returns a tuple of batched elements.  Each
    position in the input tuples is collected into a list. If the elements are tensors,
    they are stacked together using F.stack.

    The function ensures all rows in the batch have the same structure (types).

    Parameters:
    -----------
    batch : list[tuple[Any, ...]]
        A list of tuples, where each tuple contains data elements of any type.  All
        tuples must have the same length and corresponding elements must have the same
        type.

    Returns:
    --------
    tuple
        A tuple where each element is a batched version of the corresponding
        position from the input tuples. If the original elements were tensors,
        the corresponding output element will be a stacked tensor.

    Raises:
    -------
    ValueError
        If the types of elements in different rows don't match.
    IndexError
        If an empty batch is provided.

    Examples:
    ---------
    >>> # Basic types
    >>> batch = [(1, "a"), (2, "b"), (3, "c")]
    >>> default_collate_fn(batch)
    ([1, 2, 3], ['a', 'b', 'c'])

    >>> # With tensors
    >>> batch = [(Tensor([1, 2]), "a"), (Tensor([3, 4]), "b")]
    >>> result = default_collate_fn(batch)
    >>> # result[0] is a stacked tensor, result[1] is ["a", "b"]
    """
    # First, ensure the types of all elements in the batch are the same
    first_row_types = [type(elem) for elem in batch[0]]
    for row in batch:
        if [type(elem) for elem in row] != first_row_types:
            raise ValueError("All elements in the batch should have the same type")

    batches = [list() for _ in range(len(batch[0]))]

    for row in batch:
        for i, elem in enumerate(row):
            batches[i].append(elem)

    # Execute `stack` on batches of tensors.
    for i, batch in enumerate(batches):
        if isinstance(batch[0], Tensor):
            batches[i] = F.stack(batch)

    return tuple(batches)


class DataLoader[T]:
    """A data loader that loads batches of data from a dataset.

    This data loader is designed to be similar to PyTorch's DataLoader,
    providing functionality for batching, shuffling, and iterating over a dataset.

    Args:
        dataset: Dataset to load data from
        batch_size: How many samples per batch to load
        shuffle: Set to True to have the data reshuffled at every epoch
    """

    def __init__(
        self,
        dataset: Dataset[T],
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn=default_collate_fn,
    ) -> None:
        """Initialize the DataLoader.

        Args:
            dataset: Dataset to load data from
            batch_size: How many samples per batch to load
            shuffle: Set to True to have the data reshuffled at every epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

        # Validate inputs
        if batch_size <= 0:
            raise ValueError(
                f"batch_size should be a positive integer, got {batch_size}"
            )

    def __iter__(self) -> Iterator[list[T]]:
        """Create an iterator over the batches in the dataset.

        Returns:
            Iterator yielding batches from the dataset
        """
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        # Create batches
        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]
            yield self.collate_fn([self.dataset[idx] for idx in batch_indices])

    def __len__(self) -> int:
        """Get the number of batches in the dataset.

        Returns:
            Number of batches in the dataset
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
