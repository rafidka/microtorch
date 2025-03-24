"""Transforms for image processing operations.

This module provides image transformations similar to torchvision.transforms.
"""

from typing import Any, Generic, TypeVar

import numpy as np
from PIL.Image import Image

# Import your Tensor class
from microtorch.tensor import Tensor

# Define generic type variables for input and output
T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


class Transform(Generic[T_in, T_out]):
    """Base class for all transforms in microtorch.

    All transforms should subclass this class and override the `__call__` method.

    Attributes:
        None
    """

    def __init__(self) -> None:
        pass

    def __call__(self, input: T_in) -> T_out:
        """Apply the transform to the input.

        Args:
            input: Data to be transformed.

        Returns:
            Transformed data.

        Raises:
            NotImplementedError: When called on the base class.
        """
        raise NotImplementedError("Subclasses must implement __call__")


class ToTensor(Transform[np.ndarray[Any, Any] | Image, Tensor]):
    """Convert a numpy.ndarray or PIL Image to a microtorch.Tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to
    a microtorch.Tensor of shape (C x H x W) in the range [0.0, 1.0].

    If the PIL Image has mode "L", it will be converted to a tensor with a single
    channel.
    """

    def __call__(self, pic: np.ndarray[Any, Any] | Image) -> Tensor:
        """Convert a PIL Image or numpy.ndarray to tensor.

        Args:
            pic: Image to be converted to tensor.

        Returns:
            Tensor: Converted image.

        Example:
            >>> import numpy as np
            >>> from PIL import Image
            >>> img = Image.open('img.jpg')
            >>> transform = ToTensor()
            >>> tensor_img = transform(img)
        """
        # Handle PIL Image
        if isinstance(pic, Image):
            if pic.mode == "L":
                # Single-channel grayscale
                np_img = np.array(pic, dtype=np.float32)
                np_img = np_img.reshape((1, np_img.shape[0], np_img.shape[1]))
            else:
                # Multi-channel (e.g., RGB)
                np_img = np.array(pic, dtype=np.float32).transpose((2, 0, 1))
        else:
            # Handle numpy array
            if pic.ndim == 2:
                # Single-channel grayscale
                np_img = pic.reshape((1, pic.shape[0], pic.shape[1]))
            else:
                # Assume HWC format, convert to CHW
                np_img = pic.transpose((2, 0, 1))

        # Convert from [0, 255] to [0, 1]
        if np_img.max() > 1.0:
            np_img = np_img / 255.0

        # Return as microtorch Tensor
        return Tensor(np_img)


class Normalize(Transform[Tensor, Tensor]):
    """Normalize a tensor image with mean and standard deviation.

    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``microtorch.Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
        inplace: If True, performs operation in-place.
    """

    def __init__(
        self,
        mean: list[float] | tuple[float, ...] | Tensor,
        std: list[float] | tuple[float, ...] | Tensor,
        inplace: bool = False,
    ) -> None:
        """Initialize Normalize transform.

        Args:
            mean: Sequence of means for each channel.
            std: Sequence of standard deviations for each channel.
            inplace: If True, performs operation in-place.

        Example:
            >>> transform = Normalize(mean=[0.485, 0.456, 0.406],
            ...                        std=[0.229, 0.224, 0.225])
            >>> normalized_tensor = transform(tensor_img)
        """
        if isinstance(mean, Tensor):
            self.mean = mean
        else:
            self.mean = Tensor(np.array(mean, dtype=np.float32))

        if isinstance(std, Tensor):
            self.std = std
        else:
            self.std = Tensor(np.array(std, dtype=np.float32))

        self.inplace = inplace

    def __call__(self, tensor: Tensor) -> Tensor:
        """Normalize a tensor image with mean and standard deviation.

        Args:
            tensor: Tensor image to be normalized.

        Returns:
            Tensor: Normalized tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        # Check if dimensions match
        if tensor.ndim == 3:
            # Handle 3D tensor (C x H x W)
            c = tensor.shape[0]
            if len(self.mean) != c or len(self.std) != c:
                raise ValueError(
                    f"Expected mean and std to have {c} elements for a {c}-channel tensor, "
                    f"but got {len(self.mean)} and {len(self.std)} elements respectively."
                )

            # Reshape mean and std to allow broadcasting over HxW dimensions
            mean = self.mean.reshape(-1, 1, 1)
            std = self.std.reshape(-1, 1, 1)

        elif tensor.ndim == 4:
            # Handle 4D tensor (B x C x H x W)
            c = tensor.shape[1]
            if len(self.mean) != c or len(self.std) != c:
                raise ValueError(
                    f"Expected mean and std to have {c} elements for a {c}-channel tensor, "
                    f"but got {len(self.mean)} and {len(self.std)} elements respectively."
                )

            # Reshape mean and std to allow broadcasting over B, H, W dimensions
            mean = self.mean.reshape(1, -1, 1, 1)
            std = self.std.reshape(1, -1, 1, 1)
        else:
            raise ValueError(
                f"Expected tensor to be 3D (C x H x W) or 4D (B x C x H x W), "
                f"but got {tensor.ndim}D tensor instead."
            )

        # Normalize
        tensor = (tensor - mean) / std

        return tensor


class Compose[T_in, T_out](Transform[T_in, T_out]):
    """Composes several transforms together.

    Args:
        transforms: List of transforms to compose.

    Example:
        >>> transforms = Compose([
        ...     ToTensor(),
        ...     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ... ])
        >>> transformed_img = transforms(img)
    """

    def __init__(self, transforms: list[Transform[Any, Any]]) -> None:
        """Initialize Compose with a list of transforms.

        Args:
            transforms: List of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, input: T_in) -> T_out:
        """Apply transforms sequentially.

        Args:
            input: The input to transform.

        Returns:
            The transformed input.
        """
        output = input
        for t in self.transforms:
            output = t(output)
        return output  # type: ignore
