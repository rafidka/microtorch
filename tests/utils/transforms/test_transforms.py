"""Tests for transforms module."""

import numpy as np
import pytest

from microtorch.tensor import Tensor
from microtorch.utils.transforms.base import Compose, Normalize, ToTensor, Transform


class TestTransformBase:
    """Tests for the base Transform class."""

    def test_transform_not_implemented(self):
        """Test that base Transform raises NotImplementedError."""
        transform = Transform()
        with pytest.raises(NotImplementedError):
            transform("test")


class TestToTensor:
    """Tests for the ToTensor transform."""

    def test_numpy_array_2d_grayscale(self):
        """Test converting a 2D numpy array (grayscale) to tensor."""
        transform = ToTensor()
        # Create a 2D grayscale image (H x W)
        img = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
        result = transform(img)

        assert isinstance(result, Tensor)
        assert result.shape == (1, 2, 3)  # (C, H, W)
        # Check normalization to [0, 1]
        assert result.numpy().max() <= 1.0
        assert result.numpy().min() >= 0.0

    def test_numpy_array_3d_rgb(self):
        """Test converting a 3D numpy array (RGB) to tensor."""
        transform = ToTensor()
        # Create a 3D RGB image (H x W x C)
        img = np.random.randint(0, 256, (4, 5, 3), dtype=np.uint8)
        result = transform(img)

        assert isinstance(result, Tensor)
        assert result.shape == (3, 4, 5)  # (C, H, W)
        # Check normalization to [0, 1]
        assert result.numpy().max() <= 1.0
        assert result.numpy().min() >= 0.0

    def test_numpy_array_already_normalized(self):
        """Test that already normalized arrays are not double-normalized."""
        transform = ToTensor()
        # Create an already normalized array
        img = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = transform(img)

        assert isinstance(result, Tensor)
        # Values should stay in [0, 1] range
        np.testing.assert_array_almost_equal(
            result.numpy(), np.array([[[0.0, 0.5, 1.0]]])
        )


class TestToTensorPIL:
    """Tests for ToTensor with PIL images (if PIL is available)."""

    @pytest.fixture
    def pil_available(self):
        """Check if PIL is available."""
        try:
            from PIL import Image

            return Image
        except ImportError:
            pytest.skip("PIL not available")

    def test_pil_grayscale_image(self, pil_available):
        """Test converting a PIL grayscale image to tensor."""
        Image = pil_available
        transform = ToTensor()
        # Create a grayscale PIL image
        img = Image.new("L", (5, 4), color=128)
        result = transform(img)

        assert isinstance(result, Tensor)
        assert result.shape == (1, 4, 5)  # (C, H, W)
        # Check normalization: 128/255 ~ 0.502
        assert np.allclose(result.numpy(), 128 / 255, atol=0.01)

    def test_pil_rgb_image(self, pil_available):
        """Test converting a PIL RGB image to tensor."""
        Image = pil_available
        transform = ToTensor()
        # Create an RGB PIL image
        img = Image.new("RGB", (5, 4), color=(255, 128, 0))
        result = transform(img)

        assert isinstance(result, Tensor)
        assert result.shape == (3, 4, 5)  # (C, H, W)
        # Check normalization: R=1.0, G~0.502, B=0.0
        np.testing.assert_array_almost_equal(result.numpy()[0], 1.0, decimal=2)
        np.testing.assert_array_almost_equal(result.numpy()[1], 128 / 255, decimal=2)
        np.testing.assert_array_almost_equal(result.numpy()[2], 0.0, decimal=2)


class TestNormalize:
    """Tests for the Normalize transform."""

    def test_normalize_3d_tensor(self):
        """Test normalizing a 3D tensor (C x H x W)."""
        # Create a simple tensor with known values
        data = np.ones((3, 2, 2), dtype=np.float32) * 0.5
        tensor = Tensor(data)

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = Normalize(mean=mean, std=std)
        result = transform(tensor)

        # After normalization: (0.5 - 0.5) / 0.5 = 0
        expected = np.zeros((3, 2, 2), dtype=np.float32)
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_normalize_4d_tensor(self):
        """Test normalizing a 4D tensor (B x C x H x W)."""
        # Create a batch of tensors
        data = np.ones((2, 3, 4, 4), dtype=np.float32) * 0.5
        tensor = Tensor(data)

        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        transform = Normalize(mean=mean, std=std)
        result = transform(tensor)

        # After normalization: (0.5 - 0.5) / 0.25 = 0
        expected = np.zeros((2, 3, 4, 4), dtype=np.float32)
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_normalize_with_tensor_mean_std(self):
        """Test normalizing with Tensor mean and std."""
        data = np.ones((3, 2, 2), dtype=np.float32) * 0.6
        tensor = Tensor(data)

        mean = Tensor(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        std = Tensor(np.array([0.1, 0.1, 0.1], dtype=np.float32))
        transform = Normalize(mean=mean, std=std)
        result = transform(tensor)

        # After normalization: (0.6 - 0.5) / 0.1 = 1.0
        expected = np.ones((3, 2, 2), dtype=np.float32)
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_normalize_wrong_channels(self):
        """Test that normalize raises error for mismatched channels."""
        data = np.ones((3, 2, 2), dtype=np.float32)
        tensor = Tensor(data)

        # Mean/std have 2 elements but tensor has 3 channels
        mean = [0.5, 0.5]
        std = [0.5, 0.5]
        transform = Normalize(mean=mean, std=std)

        with pytest.raises(ValueError, match="Expected mean and std"):
            transform(tensor)

    def test_normalize_wrong_dimensions(self):
        """Test that normalize raises error for wrong tensor dimensions."""
        data = np.ones((2, 2), dtype=np.float32)  # 2D tensor
        tensor = Tensor(data)

        mean = [0.5]
        std = [0.5]
        transform = Normalize(mean=mean, std=std)

        with pytest.raises(ValueError, match="Expected tensor to be 3D"):
            transform(tensor)

    def test_normalize_4d_wrong_channels(self):
        """Test that normalize raises error for mismatched channels in 4D tensor."""
        data = np.ones((2, 3, 4, 4), dtype=np.float32)  # 4D tensor with 3 channels
        tensor = Tensor(data)

        # Mean/std have 2 elements but tensor has 3 channels
        mean = [0.5, 0.5]
        std = [0.5, 0.5]
        transform = Normalize(mean=mean, std=std)

        with pytest.raises(ValueError, match="Expected mean and std"):
            transform(tensor)

    def test_normalize_not_inplace(self):
        """Test that normalize creates a new tensor when inplace=False."""
        data = np.ones((3, 2, 2), dtype=np.float32) * 0.5
        tensor = Tensor(data)

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = Normalize(mean=mean, std=std, inplace=False)
        result = transform(tensor)

        # Original tensor should be unchanged
        np.testing.assert_array_almost_equal(
            tensor.numpy(), np.ones((3, 2, 2), dtype=np.float32) * 0.5
        )
        # Result should be normalized
        np.testing.assert_array_almost_equal(
            result.numpy(), np.zeros((3, 2, 2), dtype=np.float32)
        )

    def test_normalize_inplace(self):
        """Test normalize with inplace=True skips cloning."""
        data = np.ones((3, 2, 2), dtype=np.float32) * 0.5
        tensor = Tensor(data)

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        # inplace=True means the input tensor is used directly without cloning
        # (though the actual math still returns a new tensor)
        transform = Normalize(mean=mean, std=std, inplace=True)
        result = transform(tensor)

        # Result should be normalized: (0.5 - 0.5) / 0.5 = 0
        np.testing.assert_array_almost_equal(
            result.numpy(), np.zeros((3, 2, 2), dtype=np.float32)
        )


class TestToTensorInvalidInput:
    """Tests for ToTensor with invalid input types."""

    def test_to_tensor_invalid_type(self):
        """Test that ToTensor raises TypeError for invalid input types."""
        transform = ToTensor()
        with pytest.raises(TypeError, match=r"Expected numpy.ndarray or PIL Image"):
            transform([1, 2, 3])  # List is not a valid input type


class TestCompose:
    """Tests for the Compose transform."""

    def test_compose_single_transform(self):
        """Test composing a single transform."""
        transform = Compose([ToTensor()])
        img = np.random.randint(0, 256, (4, 5), dtype=np.uint8)
        result = transform(img)

        assert isinstance(result, Tensor)
        assert result.shape == (1, 4, 5)

    def test_compose_multiple_transforms(self):
        """Test composing multiple transforms."""
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        # Create a grayscale image with value 128 (0.5 after normalization)
        img = np.ones((4, 5), dtype=np.uint8) * 128
        result = transform(img)

        assert isinstance(result, Tensor)
        assert result.shape == (1, 4, 5)
        # After ToTensor: ~0.5, After Normalize: (0.5 - 0.5) / 0.5 = ~0
        # Note: 128/255 is not exactly 0.5, so we check it's close to 0
        assert np.abs(result.numpy()).max() < 0.1

    def test_compose_empty_list(self):
        """Test composing with empty list."""
        transform = Compose([])
        result = transform("test")
        assert result == "test"

    def test_compose_preserves_order(self):
        """Test that compose applies transforms in order."""

        class AddOne(Transform[float, float]):
            def __call__(self, x: float) -> float:
                return x + 1

        class MultiplyTwo(Transform[float, float]):
            def __call__(self, x: float) -> float:
                return x * 2

        # (0 + 1) * 2 = 2
        transform1 = Compose([AddOne(), MultiplyTwo()])
        assert transform1(0.0) == 2.0

        # (0 * 2) + 1 = 1
        transform2 = Compose([MultiplyTwo(), AddOne()])
        assert transform2(0.0) == 1.0
