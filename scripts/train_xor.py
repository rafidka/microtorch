#!/usr/bin/env python3
"""XOR training script for MicroTorch.

This script trains a simple neural network to learn the XOR function.
It demonstrates end-to-end training with MicroTorch and provides
visual confirmation that the framework works correctly.

Usage:
    python scripts/train_xor.py
"""

import numpy as np

from microtorch.nn import CrossEntropyLoss, Linear, Module, ReLU, Softmax
from microtorch.optim import SGD
from microtorch.tensor import Tensor


class XORNet(Module[Tensor]):
    """Simple 2-layer MLP for XOR problem."""

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


def train():
    """Train XOR network and display results."""
    print("\n" + "=" * 50)
    print("MicroTorch XOR Training Demo")
    print("=" * 50 + "\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int64)

    X_tensor = Tensor(X)
    y_tensor = Tensor(y)

    # Create model, criterion, optimizer
    model = XORNet(hidden_size=16)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.5)

    # Training
    print("Training...")
    print("-" * 50)

    num_epochs = 500
    print_every = 100

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            # Calculate accuracy
            softmax = Softmax(dim=1)
            probs = softmax(model(X_tensor))
            predictions = np.argmax(probs.numpy(), axis=1)
            accuracy = np.mean(predictions == y) * 100

            print(
                f"Epoch [{epoch + 1:4d}/{num_epochs}]  "
                f"Loss: {loss.item():.4f}  "
                f"Accuracy: {accuracy:.1f}%"
            )

    print("-" * 50)
    print("\nFinal Results:")
    print("-" * 50)

    # Final evaluation
    softmax = Softmax(dim=1)
    probs = softmax(model(X_tensor))
    predictions = np.argmax(probs.numpy(), axis=1)

    print("\nInput  | Target | Predicted | Probabilities")
    print("-" * 50)
    for i in range(4):
        prob_str = ", ".join([f"{p:.3f}" for p in probs.numpy()[i]])
        status = (
            "\033[92mOK\033[0m" if predictions[i] == y[i] else "\033[91mWRONG\033[0m"
        )
        print(
            f"{X[i]}  |   {y[i]}    |     {predictions[i]}     | [{prob_str}] {status}"
        )

    final_accuracy = np.mean(predictions == y) * 100
    print("-" * 50)
    print(f"\nFinal Accuracy: {final_accuracy:.1f}%")

    if final_accuracy == 100:
        print("\n\033[92mXOR learning successful!\033[0m")
    else:
        print(f"\n\033[93mPartial success - {final_accuracy:.0f}% accuracy\033[0m")

    return final_accuracy == 100


def main():
    """Main entry point."""
    success = train()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
