# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MicroTorch is an educational deep learning framework inspired by PyTorch, implementing core deep learning concepts from scratch using NumPy. The codebase closely follows PyTorch's style and architecture for educational purposes.

## Development Commands

### Environment Setup
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install with development dependencies
uv sync --all-extras
```

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/tensor/test_tensor.py

# Run specific test
uv run pytest tests/tensor/test_tensor.py::test_tensor_initialization

# Run tests for a specific module
uv run pytest tests/nn/
```

### Code Quality
```bash
# Run linter
uv run ruff check src/ tests/

# Run linter with auto-fix
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/

# Run type checker
uv run pyright
```

## Architecture Overview

The framework is organized into several key modules that mirror PyTorch's structure:

### Core Components

1. **Tensor Module** (`src/microtorch/tensor/`)
   - `Tensor` class: Core tensor implementation with automatic differentiation
   - Supports backward propagation through `_backward` functions
   - Gradient tracking with `requires_grad` flag

2. **Neural Network Module** (`src/microtorch/nn/`)
   - `Module`: Base class for all neural network layers
   - `Parameter`: Wrapper for trainable tensors
   - Functional API in `nn/functional/` for operations
   - Layer implementations in `nn/modules/`

3. **Optimizer Module** (`src/microtorch/optim/`)
   - Optimization algorithms for training
   - Follows PyTorch's optimizer interface

4. **Utils Module** (`src/microtorch/utils/`)
   - Data utilities including `Dataset` and `DataLoader`
   - Transform utilities for data preprocessing

### Key Design Patterns

- **Automatic Differentiation**: Each tensor operation stores backward functions for gradient computation
- **Module System**: Neural network layers inherit from `Module` base class with automatic parameter registration
- **Parameter Management**: Parameters are automatically tracked when assigned as module attributes
- **Functional API**: Operations available both as tensor methods and functional API

### Testing Requirements

- Minimum code coverage: 98% (configured in pyproject.toml)
- Tests organized to mirror source structure
- Coverage reports generated in `coverage_html/` directory

### Type Checking

- Strict type checking enabled with pyright
- Uses Python 3.12+ type hints including generics
- Type stubs provided via `py.typed` markers

### Code Style

- Line length: 88 characters (Black-compatible)
- Google docstring convention
- Import sorting with isort configuration
- Complexity limit: 15 (McCabe)