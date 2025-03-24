class Dataset[T]:
    """Base class for all datasets in microtorch.

    All datasets should subclass this class and override the `__getitem__` and
    `__len__` methods.
    """

    def __init__(self) -> None:
        pass

    def __getitem__(self, index: int) -> T:
        """Get the item at the given index.

        Args:
            index: Index of the item to get.

        Returns:
            Item at the given index.
        """
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Length of the dataset.
        """
        raise NotImplementedError("Subclasses must implement __len__")
