"""End-to-end training integration tests.

These tests verify that all components of the framework work together
correctly in realistic training scenarios.
"""

import numpy as np

import microtorch
from microtorch.nn import CrossEntropyLoss, Linear, Module, ReLU, Softmax
from microtorch.optim import SGD
from microtorch.tensor import Tensor, functional as F
from microtorch.utils.data import DataLoader, Dataset

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


class XORDataset(Dataset[tuple[Tensor, Tensor]]):
    """XOR dataset for testing."""

    def __init__(self):
        # XOR inputs and outputs
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        self.y = np.array([0, 1, 1, 0], dtype=np.int64)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return Tensor(self.X[index]), Tensor(np.array([self.y[index]]))

    def __len__(self) -> int:
        return len(self.X)


class XORModel(Module[Tensor]):
    """Simple MLP for XOR problem."""

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.fc1 = Linear(2, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class LinearRegressionModel(Module[Tensor]):
    """Simple linear model for regression."""

    def __init__(self):
        super().__init__()
        self.linear = Linear(1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class TestXORConvergence:
    """Test that a model can learn the XOR function."""

    def test_xor_convergence_basic(self):
        """Test that XOR model converges to high accuracy."""
        microtorch.manual_seed(42)

        model = XORModel(hidden_size=8)
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.5)

        # XOR data
        inputs = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
        targets = Tensor(np.array([0, 1, 1, 0]))

        initial_loss = None
        final_loss = None

        # Train for enough epochs
        for epoch in range(500):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if epoch == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Verify loss decreased significantly
        assert initial_loss is not None
        assert final_loss is not None
        assert final_loss < initial_loss * 0.1, (
            f"Loss didn't decrease enough: {initial_loss} -> {final_loss}"
        )

        # Check accuracy
        with_softmax = Softmax(dim=1)
        probs = with_softmax(model(inputs))
        predictions = np.argmax(probs.numpy(), axis=1)
        expected = np.array([0, 1, 1, 0])
        accuracy = np.mean(predictions == expected)

        assert accuracy >= 0.75, f"Accuracy too low: {accuracy}"

    def test_xor_convergence_with_dataloader(self):
        """Test XOR with DataLoader integration."""
        microtorch.manual_seed(42)

        dataset = XORDataset()
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = XORModel(hidden_size=16)
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.5)

        losses = []
        for _ in range(200):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                # Squeeze batch_y from (4, 1) to (4,)
                loss = criterion(outputs, batch_y.reshape((-1,)))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss)

        # Verify loss trend is decreasing
        assert losses[-1] < losses[0] * 0.5, "Loss should decrease over training"


class TestLinearRegression:
    """Test linear regression convergence."""

    def test_linear_regression_convergence(self):
        """Test that linear model can fit a line."""
        rng = np.random.default_rng(42)
        microtorch.manual_seed(42)  # For weight initialization

        # Generate data: y = 2x + 1 + noise
        x_data = np.linspace(-1, 1, 20).reshape(-1, 1).astype(np.float32)
        y_true = 2 * x_data + 1
        y_noise = y_true + rng.standard_normal(y_true.shape).astype(np.float32) * 0.1

        x_tensor = Tensor(x_data)
        y_tensor = Tensor(y_noise)

        model = LinearRegressionModel()
        optimizer = SGD(model.parameters(), lr=0.1)

        # MSE loss
        def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
            diff = pred - target
            return F.sum(diff * diff) / Tensor(np.array(pred.shape[0]))

        # Capture initial loss
        initial_pred = model(x_tensor)
        initial_loss = mse_loss(initial_pred, y_tensor).item()

        for _ in range(200):
            optimizer.zero_grad()
            pred = model(x_tensor)
            loss = mse_loss(pred, y_tensor)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Verify loss decreased
        assert initial_loss is not None
        assert final_loss < initial_loss * 0.1, (
            f"Loss didn't decrease enough: {initial_loss} -> {final_loss}"
        )

        # Check that parameters are close to true values (w=2, b=1)
        # Note: Due to noise and limited epochs, we just check rough convergence
        weight = model.linear.weight.numpy().flatten()[0]
        bias = model.linear.bias.numpy().flatten()[0]

        assert abs(weight - 2.0) < 0.5, f"Weight should be ~2, got {weight}"
        assert abs(bias - 1.0) < 0.5, f"Bias should be ~1, got {bias}"


class TestTrainingLoopComponents:
    """Test individual training loop components work together."""

    def test_forward_backward_step_cycle(self):
        """Test a single forward-backward-step cycle."""
        rng = np.random.default_rng(42)
        microtorch.manual_seed(42)  # For weight initialization

        model = Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        x = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
        y = Tensor(np.array([0, 1, 0]))

        # Get initial parameters
        initial_weight = model.weight.numpy().copy()
        initial_bias = model.bias.numpy().copy()

        # Forward
        output = model(x)
        assert output.shape == (3, 2)

        # Loss
        loss = criterion(output, y)
        assert loss.shape == ()  # Scalar

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        assert model.weight.grad is not None
        assert model.bias.grad is not None
        assert not np.allclose(model.weight.grad, 0)

        # Step
        optimizer.step()

        # Verify parameters changed
        assert not np.allclose(model.weight.numpy(), initial_weight)
        assert not np.allclose(model.bias.numpy(), initial_bias)

    def test_multiple_batches(self):
        """Test training over multiple batches."""
        rng = np.random.default_rng(42)
        microtorch.manual_seed(42)  # For weight initialization

        model = Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        losses = []
        for _ in range(10):
            x = Tensor(rng.standard_normal((8, 4)).astype(np.float32))
            y = Tensor(rng.integers(0, 2, size=8))

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Just verify no crashes and losses are finite
        assert all(np.isfinite(loss_val) for loss_val in losses)


class TestGradientAccumulation:
    """Test gradient accumulation behavior."""

    def test_gradients_accumulate(self):
        """Test that gradients accumulate without zero_grad."""
        rng = np.random.default_rng(42)
        microtorch.manual_seed(42)  # For weight initialization

        model = Linear(4, 2)
        x = Tensor(rng.standard_normal((3, 4)).astype(np.float32), requires_grad=True)

        # First forward-backward
        out1 = F.sum(model(x))
        out1.backward()
        grad_after_first = model.weight.grad.copy()

        # Second forward-backward without zero_grad
        out2 = F.sum(model(x))
        out2.backward()
        grad_after_second = model.weight.grad.copy()

        # Gradients should have accumulated (doubled)
        np.testing.assert_array_almost_equal(
            grad_after_second,
            grad_after_first * 2,
            err_msg="Gradients should accumulate",
        )

    def test_zero_grad_resets(self):
        """Test that zero_grad resets gradients."""
        rng = np.random.default_rng(42)
        microtorch.manual_seed(42)  # For weight initialization

        model = Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        x = Tensor(rng.standard_normal((3, 4)).astype(np.float32))

        # First forward-backward
        out1 = F.sum(model(x))
        out1.backward()
        grad_after_first = model.weight.grad.copy()

        # Zero gradients
        optimizer.zero_grad()

        # Verify gradients are zero
        np.testing.assert_array_equal(
            model.weight.grad, np.zeros_like(model.weight.grad)
        )

        # Second forward-backward
        out2 = F.sum(model(x))
        out2.backward()
        grad_after_second = model.weight.grad.copy()

        # Gradients should be same as first (not accumulated)
        np.testing.assert_array_almost_equal(
            grad_after_second,
            grad_after_first,
            err_msg="Gradients should be same after zero_grad",
        )


class TestDataLoaderIntegration:
    """Test DataLoader integration with training."""

    def test_dataloader_epoch_iteration(self):
        """Test iterating through DataLoader multiple epochs."""
        dataset = XORDataset()
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        all_batches = []
        for _ in range(3):
            epoch_batches = []
            for batch_x, batch_y in dataloader:
                epoch_batches.append((batch_x.numpy(), batch_y.numpy()))
            all_batches.append(epoch_batches)

        # Should have 2 batches per epoch (4 samples / 2 batch_size)
        assert all(len(epoch) == 2 for epoch in all_batches)

        # With shuffle, order should differ between epochs (probabilistically)
        # We just verify the data is correct, not the order
        for epoch_batches in all_batches:
            all_x = np.concatenate([b[0] for b in epoch_batches])
            all_y = np.concatenate([b[1] for b in epoch_batches])

            # All XOR inputs should be present
            assert len(all_x) == 4
            assert len(all_y) == 4

    def test_custom_dataset_with_training(self):
        """Test a custom dataset works with the training loop."""

        class SimpleDataset(Dataset[tuple[Tensor, Tensor]]):
            def __init__(self, size: int = 100):
                rng = np.random.default_rng(42)
                self.data = rng.standard_normal((size, 4)).astype(np.float32)
                self.labels = (self.data.sum(axis=1) > 0).astype(np.int64)

            def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
                return Tensor(self.data[index]), Tensor(np.array([self.labels[index]]))

            def __len__(self) -> int:
                return len(self.data)

        dataset = SimpleDataset(100)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

        model = Linear(4, 2)
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.1)

        # Train for a few epochs
        for _ in range(5):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y.reshape((-1,)))
                loss.backward()
                optimizer.step()

        # Verify training completed without errors
        # Final loss should be finite
        assert np.isfinite(loss.item())
