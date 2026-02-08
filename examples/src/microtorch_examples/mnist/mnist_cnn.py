"""MNIST classification using a Convolutional Neural Network."""

from typing import Literal, override

import datasets as hf_datasets  # type: ignore
from PIL.Image import Image

from microtorch import nn, optim
from microtorch.tensor import Tensor
from microtorch.utils import transforms
from microtorch.utils.data import DataLoader, Dataset


class MnistCNN(nn.Module[Tensor]):
    """
    Simple CNN for MNIST classification.

    Architecture:
        Conv2d(1, 32, 3, padding=1) → ReLU → (N, 32, 28, 28)
        Conv2d(32, 64, 3, stride=2, padding=1) → ReLU → (N, 64, 14, 14)
        Conv2d(64, 64, 3, stride=2, padding=1) → ReLU → (N, 64, 7, 7)
        Flatten → (N, 64*7*7)
        Linear(64*7*7, 10) → (N, 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Activation
        self.relu = nn.ReLU()

        # Fully connected layer
        # After conv layers: 28 → 28 → 14 → 7, so feature map is 64 * 7 * 7
        self.fc = nn.Linear(64 * 7 * 7, num_classes)
        self.softmax = nn.Softmax(dim=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 1, 28, 28)
        x = self.relu(self.conv1(x))  # (N, 32, 28, 28)
        x = self.relu(self.conv2(x))  # (N, 64, 14, 14)
        x = self.relu(self.conv3(x))  # (N, 64, 7, 7)

        # Flatten: (N, 64, 7, 7) → (N, 64*7*7)
        x = x.reshape((x.shape[0], -1))

        x = self.fc(x)  # (N, 10)
        return x  # self.softmax(x)


class MnistDataset(Dataset[tuple[Tensor, Tensor]]):
    """MNIST dataset wrapper using HuggingFace datasets."""

    def __init__(
        self,
        split: Literal["train", "test"],
        transform: transforms.Transform[Image, Tensor],
    ) -> None:
        self.dataset: hf_datasets.Dataset = hf_datasets.load_dataset(  # type: ignore
            "mnist", split=split
        )
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image: Image = self.dataset[index]["image"]
        label: int = self.dataset[index]["label"]
        tensor = self.transform(image)
        return tensor, Tensor(label)

    def __len__(self) -> int:
        return len(self.dataset)


def main() -> None:
    """Train a CNN on MNIST."""
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 3

    # Data transforms
    transform = transforms.Compose[Image, Tensor](
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Datasets and loaders
    train_dataset = MnistDataset(split="train", transform=transform)
    test_dataset = MnistDataset(split="test", transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = MnistCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting CNN training on MNIST...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for idx, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Batch [{idx}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed, Avg Loss: {avg_loss:.4f}")

    # Evaluation
    print("\nEvaluating on test set...")
    correct = 0
    total = 0

    for images, labels in test_loader:
        outputs = model(images)
        # Get predicted class (argmax)
        predictions = outputs.numpy().argmax(axis=1)
        targets = labels.numpy().flatten().astype(int)

        correct += (predictions == targets).sum()
        total += len(targets)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
