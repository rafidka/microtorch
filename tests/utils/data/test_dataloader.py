import pytest

import microtorch
from microtorch.tensor.tensor import Tensor
from microtorch.utils.data.dataloader import default_collate_fn


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
