from typing import Literal

import datasets as hf_datasets  # type: ignore
from PIL.Image import Image

from microtorch import nn, optim
from microtorch.tensor import Tensor
from microtorch.utils import transforms
from microtorch.utils.data import DataLoader, Dataset


class MnistModel(nn.Module[Tensor]):
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape((-1, self.input_size))  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


class MnistDataset(Dataset[tuple[Tensor, Tensor]]):
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
        return tensor, Tensor([label])

    def __len__(self) -> int:
        return len(self.dataset)


model = MnistModel()

# Hyperparameters
batch_size = 128
learning_rate = 0.01
num_epochs = 5

# MNIST dataset
transform = transforms.Compose[Image, Tensor](
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


train_dataset = MnistDataset(split="train", transform=transform)
test_dataset = MnistDataset(split="test", transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for batch in train_loader:
    images, labels = batch
    break

# Model, Loss, and Optimizer
model = MnistModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], [{idx}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# # Evaluation
# correct = 0
# total = 0
# for images, labels in test_loader:
#     outputs = model(images)
#     _, predicted = torch.max(outputs, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()
#
# print(f"Accuracy on test set: {100 * correct / total:.2f}%")
#
