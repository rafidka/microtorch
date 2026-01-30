import pytest

import microtorch
from microtorch.tensor.tensor import Tensor
from microtorch.utils.data.dataloader import DataLoader, default_collate_fn
from microtorch.utils.data.dataset import Dataset


def test_default_collate_fn_ints():
    batch = [(1, 2), (3, 4), (5, 6)]
    collated = default_collate_fn(batch)
    assert collated == ([1, 3, 5], [2, 4, 6])


def test_default_collate_fn_strs():
    batch = [("a", "b"), ("c", "d"), ("e", "f")]
    collated = default_collate_fn(batch)
    assert collated == (["a", "c", "e"], ["b", "d", "f"])


def test_default_collate_fn_tensors():
    batch = [
        (Tensor([1, 2]), Tensor([3, 4])),
        (Tensor([5, 6]), Tensor([7, 8])),
    ]
    collated: tuple[Tensor, Tensor] = default_collate_fn(batch)  # type: ignore
    assert len(collated) == 2
    assert microtorch.equal(
        collated[0],
        Tensor(
            [
                [1, 2],
                [5, 6],
            ]
        ),
    )
    assert microtorch.equal(
        collated[1],
        Tensor(
            [
                [3, 4],
                [7, 8],
            ]
        ),
    )


def test_default_collate_fn_tensors_and_labels():
    batch = [
        (Tensor([1, 2]), Tensor([3, 4]), "a"),
        (Tensor([5, 6]), Tensor([7, 8]), "b"),
    ]
    collated: tuple[Tensor, Tensor, list[str]] = default_collate_fn(batch)  # type: ignore
    assert len(collated) == 3
    assert microtorch.equal(
        collated[0],
        Tensor(
            [
                [1, 2],
                [5, 6],
            ]
        ),
    )
    assert microtorch.equal(
        collated[1],
        Tensor(
            [
                [3, 4],
                [7, 8],
            ]
        ),
    )
    assert collated[2] == ["a", "b"]


def test_default_collate_fn_empty_batch():
    with pytest.raises(IndexError):
        default_collate_fn([])


def test_default_collate_fn_inconsistent_types():
    """Test behavior when the batch has inconsistent types."""
    batch = [
        (1, "a"),
        (2, 3),  # Second element is an int instead of a string
    ]
    with pytest.raises(
        ValueError, match="All elements in the batch should have the same type"
    ):
        default_collate_fn(batch)


def test_default_collate_fn_inconsistent_row_lengths():
    """Test behavior when rows have different lengths."""
    batch = [
        (1, "a", 0.5),
        (2, "b"),  # Missing the third element
    ]
    with pytest.raises(
        ValueError, match="All elements in the batch should have the same type"
    ):
        default_collate_fn(batch)


def test_default_collate_fn_single_element_batch():
    """Test behavior with a batch containing a single element."""
    batch = [
        (1, "a", 0.5),
    ]
    assert default_collate_fn(batch) == ([1], ["a"], [0.5])


# Tests for DataLoader class


class SimpleDataset(Dataset[tuple[int, str]]):
    """A simple dataset for testing."""

    def __init__(self, data: list[tuple[int, str]]):
        self.data = data

    def __getitem__(self, index: int) -> tuple[int, str]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def test_dataloader_basic():
    """Test basic DataLoader functionality."""
    dataset = SimpleDataset([(1, "a"), (2, "b"), (3, "c"), (4, "d")])
    loader = DataLoader(dataset, batch_size=2)

    batches = list(loader)
    assert len(batches) == 2
    assert batches[0] == ([1, 2], ["a", "b"])
    assert batches[1] == ([3, 4], ["c", "d"])


def test_dataloader_len():
    """Test DataLoader __len__ method."""
    dataset = SimpleDataset([(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")])
    loader = DataLoader(dataset, batch_size=2)
    assert len(loader) == 3  # 5 items / 2 batch_size = 3 batches (ceil division)


def test_dataloader_shuffle():
    """Test DataLoader shuffle functionality."""
    dataset = SimpleDataset([(i, str(i)) for i in range(100)])
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Collect all values and check they're all present
    all_values = []
    for batch in loader:
        all_values.extend(batch[0])

    assert sorted(all_values) == list(range(100))


def test_dataloader_invalid_batch_size():
    """Test DataLoader with invalid batch size."""
    dataset = SimpleDataset([(1, "a")])
    with pytest.raises(ValueError, match="batch_size should be a positive integer"):
        DataLoader(dataset, batch_size=0)

    with pytest.raises(ValueError, match="batch_size should be a positive integer"):
        DataLoader(dataset, batch_size=-1)


def test_dataloader_custom_collate_fn():
    """Test DataLoader with custom collate function."""

    def custom_collate(batch: list) -> list:
        return [item[0] * 2 for item in batch]

    dataset = SimpleDataset([(1, "a"), (2, "b"), (3, "c")])
    loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)

    batches = list(loader)
    assert batches[0] == [2, 4]  # 1*2, 2*2
    assert batches[1] == [6]  # 3*2


def test_dataloader_batch_size_larger_than_dataset():
    """Test DataLoader when batch size is larger than dataset."""
    dataset = SimpleDataset([(1, "a"), (2, "b")])
    loader = DataLoader(dataset, batch_size=10)

    batches = list(loader)
    assert len(batches) == 1
    assert batches[0] == ([1, 2], ["a", "b"])
