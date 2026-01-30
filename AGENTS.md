# AGENTS.md

Guidelines for AI coding agents working in the MicroTorch repository.

## Project Overview

MicroTorch is an educational deep learning framework inspired by PyTorch, implementing core concepts from scratch using NumPy. The codebase follows PyTorch's style and API conventions.

## Build & Development Commands

### Environment Setup
```bash
uv sync                    # Install all dependencies
uv sync --all-extras       # Install with all optional dependencies
```

### Running Tests
```bash
# Full test suite with coverage (default)
uv run pytest

# Single test file
uv run pytest tests/tensor/test_functional.py

# Single test function
uv run pytest tests/tensor/test_functional.py::test_add

# Single test class
uv run pytest tests/tensor/test_functional.py::TestMatmul

# Single test method in a class
uv run pytest tests/tensor/test_functional.py::TestMatmul::test_matmul_2d

# Run without coverage (faster)
uv run pytest --no-cov tests/tensor/test_functional.py

# Verbose output
uv run pytest -v tests/tensor/test_functional.py

# Stop on first failure
uv run pytest -x
```

### Linting & Type Checking
```bash
# Type checking (source only - tests excluded)
uv run pyright src/

# Linting
uv run ruff check src/ tests/

# Lint with auto-fix
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/
```

## Code Style Guidelines

### Import Organization
Imports must be in three groups, separated by blank lines:
```python
# Python imports (stdlib)
from collections.abc import Callable
from typing import Any, Self, override

# 3rd-party imports
import numpy as np

# Local imports
from microtorch.tensor import Tensor, functional as F
```

Use `functional as F` alias (PyTorch convention). Imports are sorted by isort.

### Type Annotations
- **All functions require return type annotations** (even `-> None`)
- Use `Self` for in-place methods returning `self`
- Use `@override` decorator for methods overriding parent class
- Use forward references (`"Tensor"`) for self-referential types
- Generics use Python 3.12+ syntax: `class Module[T]:`

```python
def __init__(self, data: np.ndarray[Any, Any]) -> None:
    ...

def __iadd__(self, other: "Tensor") -> Self:
    self._move(F.add(self, other))
    return self

@override
def forward(self, x: Tensor) -> Tensor:
    return F.relu(x)
```

### Docstrings
- Use Google docstring convention
- Required for public classes and functions in `src/`
- Not required in tests, scripts, or notebooks

### Naming Conventions
- Classes: `PascalCase` (e.g., `CrossEntropyLoss`)
- Functions/variables: `snake_case` (e.g., `requires_grad`)
- Protected attributes: `_prefix` (e.g., `_data`, `_backward`)
- Constants: `UPPER_CASE`
- PyTorch-style names are acceptable: `input`, `forward`, `F.relu`

### Error Handling
- Use descriptive error messages with f-strings
- Include actual vs expected values in messages
- Raise specific exceptions (`ValueError`, `TypeError`)

### NumPy Usage
- Use modern Generator API: `np.random.default_rng().standard_normal(shape)`
- Avoid legacy API: `np.random.randn()`, `np.random.seed()`

### Protected Member Access
- Use `_data` directly within the package (not `numpy()` which copies)
- Add pyright ignore for legitimate internal access:
  ```python
  return bool(np.equal(a._data, b._data).all())  # pyright: ignore[reportPrivateUsage]
  ```

## Architecture Patterns

### Tensor Operations (functional.py)
Each operation follows this pattern:
```python
def operation(a: "tensor.Tensor", ...) -> "tensor.Tensor":
    out = tensor.Tensor(result_data, requires_grad=...)
    
    def _backward():
        if a.requires_grad:
            a.grad += gradient_computation
    
    out._backward = _backward
    out._prev = [a, ...]
    out._op = "operation_name"
    out._is_leaf = False
    return out
```

### Neural Network Modules
- Inherit from `Module[T]` where `T` is the return type of `forward()`
- Parameters auto-register when assigned as attributes
- Implement `forward()` with `@override` decorator

## Architecture Overview

The framework mirrors PyTorch's structure:

1. **Tensor Module** (`src/microtorch/tensor/`)
   - `Tensor` class with automatic differentiation
   - Backward propagation via `_backward` functions
   - Gradient tracking with `requires_grad` flag

2. **Neural Network Module** (`src/microtorch/nn/`)
   - `Module`: Base class for all layers
   - `Parameter`: Wrapper for trainable tensors
   - Functional API in `tensor/functional.py`

3. **Optimizer Module** (`src/microtorch/optim/`)
   - SGD and base Optimizer class
   - Follows PyTorch's optimizer interface

4. **Utils Module** (`src/microtorch/utils/`)
   - `Dataset` and `DataLoader` for data handling
   - Transform utilities for preprocessing

## Testing Requirements

- **Minimum coverage: 98%** (enforced by pytest-cov)
- Tests mirror source structure: `src/microtorch/tensor/` → `tests/tensor/`
- Use numerical gradient checking for new operations
- Include edge cases: empty tensors, scalars, broadcasting

## Files to Ignore

These are auto-generated or environment-specific:
- `coverage_html/`, `coverage.xml`
- `.venv/`, `__pycache__/`
- `uv.lock` (commit this, but don't manually edit)
- `*.egg-info/`
