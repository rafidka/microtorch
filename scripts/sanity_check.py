#!/usr/bin/env python3
"""Sanity check script for MicroTorch.

This script performs quick smoke tests to verify that all major components
of MicroTorch are working correctly. Run this before commits to catch
obvious regressions.

Usage:
    python scripts/sanity_check.py
"""

import sys
import time

import numpy as np


def print_status(name: str, passed: bool, duration: float) -> None:
    """Print test status."""
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"  [{status}] {name} ({duration:.3f}s)")


def run_check(name: str, check_fn) -> bool:
    """Run a single check and report results."""
    start = time.time()
    try:
        check_fn()
        duration = time.time() - start
        print_status(name, True, duration)
        return True
    except Exception as e:
        duration = time.time() - start
        print_status(name, False, duration)
        print(f"         Error: {e}")
        return False


def check_tensor_creation():
    """Check tensor creation."""
    from microtorch.tensor import Tensor

    t = Tensor([1, 2, 3])
    assert t.shape == (3,)
    assert len(t) == 3

    t_grad = Tensor([1, 2, 3], requires_grad=True)
    assert t_grad.requires_grad
    assert t_grad.grad is not None


def check_tensor_operations():
    """Check basic tensor operations."""
    from microtorch.tensor import Tensor

    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])

    # Arithmetic
    _ = a + b
    _ = a - b
    _ = a * b
    _ = a / b
    _ = -a

    # Matmul
    m1 = Tensor([[1, 2], [3, 4]])
    m2 = Tensor([[5, 6], [7, 8]])
    _ = m1 @ m2


def check_autograd():
    """Check autograd functionality."""
    from microtorch.tensor import Tensor, functional as F

    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    c = a + b
    d = c * a
    e = F.sum(d)
    e.backward()

    assert a.grad is not None
    assert b.grad is not None


def check_functional_api():
    """Check functional API."""
    from microtorch.tensor import Tensor, functional as F

    a = Tensor([[1, 2], [3, 4]], requires_grad=True)

    _ = F.sum(a)
    _ = F.max(a)
    _ = F.relu(a)
    _ = F.sin(a)
    _ = F.cos(a)
    _ = F.exp(a * Tensor(np.array(0.1)))  # Scale to avoid overflow
    _ = F.reshape(a, (4,))


def check_neural_network_modules():
    """Check neural network modules."""
    from microtorch.nn import CrossEntropyLoss, Linear, ReLU, Softmax
    from microtorch.tensor import Tensor

    # Linear layer
    linear = Linear(4, 2)
    x = Tensor(np.random.randn(3, 4))
    y = linear(x)
    assert y.shape == (3, 2)

    # ReLU
    relu = ReLU()
    _ = relu(Tensor([-1, 0, 1]))

    # Softmax
    softmax = Softmax(dim=1)
    _ = softmax(Tensor([[1, 2, 3], [4, 5, 6]]))

    # CrossEntropyLoss
    criterion = CrossEntropyLoss()
    logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target = Tensor([2])
    loss = criterion(logits, target)
    loss.backward()


def check_optimizer():
    """Check optimizer functionality."""
    from microtorch.nn import Linear
    from microtorch.optim import SGD
    from microtorch.tensor import Tensor, functional as F

    linear = Linear(4, 2)
    optimizer = SGD(linear.parameters(), lr=0.01)

    x = Tensor(np.random.randn(3, 4))
    y = linear(x)
    loss = F.sum(y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def check_data_utilities():
    """Check data utilities."""
    from microtorch.tensor import Tensor
    from microtorch.utils.data import DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __getitem__(self, idx):
            return (Tensor([idx]), Tensor([idx * 2]))

        def __len__(self):
            return 10

    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=2)

    for batch in dataloader:
        assert len(batch) == 2
        break


def check_transforms():
    """Check transform utilities."""
    from microtorch.utils.transforms import Compose, Normalize, ToTensor

    # ToTensor
    to_tensor = ToTensor()
    img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    tensor = to_tensor(img)
    assert tensor.shape == (1, 28, 28)

    # Normalize
    normalize = Normalize(mean=[0.5], std=[0.5])
    normalized = normalize(tensor)
    assert normalized.shape == (1, 28, 28)

    # Compose
    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    result = transform(img)
    assert result.shape == (1, 28, 28)


def main():
    """Run all sanity checks."""
    print("\n" + "=" * 50)
    print("MicroTorch Sanity Check")
    print("=" * 50 + "\n")

    checks = [
        ("Tensor Creation", check_tensor_creation),
        ("Tensor Operations", check_tensor_operations),
        ("Autograd", check_autograd),
        ("Functional API", check_functional_api),
        ("Neural Network Modules", check_neural_network_modules),
        ("Optimizer", check_optimizer),
        ("Data Utilities", check_data_utilities),
        ("Transforms", check_transforms),
    ]

    total_start = time.time()
    passed = 0
    failed = 0

    for name, check_fn in checks:
        if run_check(name, check_fn):
            passed += 1
        else:
            failed += 1

    total_duration = time.time() - total_start

    print("\n" + "-" * 50)
    print(f"Total: {passed} passed, {failed} failed ({total_duration:.3f}s)")

    if failed > 0:
        print("\n\033[91mSanity check FAILED!\033[0m")
        sys.exit(1)
    else:
        print("\n\033[92mAll sanity checks PASSED!\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
