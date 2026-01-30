# MicroTorch

A minimal deep learning framework built from scratch for educational purposes. MicroTorch is heavily inspired by PyTorch and closely follows its API design, making it an excellent resource for understanding how deep learning frameworks work under the hood.

## Features

- **Automatic Differentiation**: Full autograd support with backpropagation
- **Neural Network Modules**: `Linear`, `ReLU`, `Softmax`, `CrossEntropyLoss`
- **Optimizers**: SGD with extensible optimizer base class
- **Data Utilities**: `Dataset`, `DataLoader` with batching and shuffling
- **Transforms**: `ToTensor`, `Normalize`, `Compose`
- **NumPy Backend**: All tensor operations implemented using NumPy
- **PyTorch-style API**: Familiar interface for PyTorch users

## Installation

Requires Python 3.12+.

```bash
# Clone the repository
git clone https://github.com/yourusername/microtorch.git
cd microtorch

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

### Basic Tensor Operations

```python
from microtorch.tensor import Tensor, functional as F

# Create tensors with gradient tracking
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

# Operations
c = a + b
d = F.matmul(a, b)
e = F.sum(d)

# Backpropagation
e.backward()

print(a.grad)  # Gradients with respect to a
```

### Training a Neural Network

```python
import numpy as np
from microtorch.tensor import Tensor
from microtorch.nn import Linear, ReLU, CrossEntropyLoss, Module
from microtorch.optim import SGD

# Define a simple model
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 8)
        self.relu = ReLU()
        self.fc2 = Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# XOR dataset
X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
y = Tensor(np.array([0, 1, 1, 0]))

# Training
model = MLP()
optimizer = SGD(model.parameters(), lr=0.1)
criterion = CrossEntropyLoss()

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Using DataLoader

```python
from microtorch.tensor import Tensor
from microtorch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.X = np.random.randn(100, 4).astype(np.float32)
        self.y = (self.X.sum(axis=1) > 0).astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return Tensor(self.X[idx]), Tensor(np.array(self.y[idx]))

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_x, batch_y in dataloader:
    # Training loop here
    pass
```

## Architecture

```
src/microtorch/
├── tensor/
│   ├── tensor.py       # Core Tensor class with autograd
│   ├── functional.py   # Differentiable operations (add, matmul, relu, etc.)
│   └── backprop.py     # Backward pass implementation
├── nn/
│   └── modules/        # Neural network layers (Linear, ReLU, etc.)
├── optim/              # Optimizers (SGD)
└── utils/
    ├── data/           # Dataset and DataLoader
    └── transforms/     # Data transforms (ToTensor, Normalize)
```

## Development

### Setup

```bash
uv sync  # Install all dependencies including dev tools
```

### Testing

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/tensor/test_functional.py

# Run without coverage (faster)
uv run pytest --no-cov
```

### Code Quality

```bash
# Type checking
uv run pyright src/

# Linting
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/
```

## Requirements

- Python >= 3.12
- NumPy >= 2.2.3

## Educational Resources

This project is designed for learning. Key concepts to explore:

1. **Automatic Differentiation**: See `tensor/functional.py` for how each operation defines its backward pass
2. **Computational Graphs**: Check `tensor/backprop.py` for topological sorting and gradient propagation
3. **Module System**: Explore `nn/modules/module.py` for automatic parameter registration
4. **Optimizers**: Study `optim/sgd.py` for gradient descent implementation

## License

MIT License

## Acknowledgments

Inspired by [PyTorch](https://pytorch.org/) and educational projects like [micrograd](https://github.com/karpathy/micrograd).
