# Code Analysis Report: Bugs and Quality Issues

## 🐛 Critical Bugs

### 1. Bug in `tensor.py` line 51: Typo in `_move()` method
```python
self._topo_order = other._is_leaf  # Should be other._topo_order
```
**Issue**: This assigns a boolean value instead of the topological order integer.
**File**: `src/microtorch/tensor/tensor.py:51`

### 2. Bug in `functional.py` line 294: Wrong operation name
```python
out._op = "sin"  # Should be "cos" 
```
**Issue**: The `cos()` function incorrectly labels its operation as "sin".
**File**: `src/microtorch/tensor/functional.py:294`

### 3. Bug in `functional.py` line 99: Incorrect gradient sign
```python
b.grad -= delta_grad.reshape(b.grad.shape)  # Should be += 
```
**Issue**: Subtraction operation's backward pass incorrectly subtracts gradient for `b`.
**File**: `src/microtorch/tensor/functional.py:99`

## ⚠️ Potential Issues

### 4. Missing assertions in mul/div backward passes
**Issue**: Lines 159, 164 in `functional.py` don't check if `out.grad` is not None before using it.
**File**: `src/microtorch/tensor/functional.py:159,164`

### 5. Type inconsistency in DataLoader
**Issue**: `dataloader.py` line 110 return type says `Iterator[list[T]]` but `collate_fn` returns tuples, not lists.
**File**: `src/microtorch/utils/data/dataloader.py:110`

### 6. Missing gradient initialization check
**Issue**: Multiple functions in `functional.py` assume gradients exist but don't verify before operations.
**Files**: Various locations in `src/microtorch/tensor/functional.py`

### 7. Inefficient broadcasting dimension calculation
**Issue**: The padding with `-1` in `utils.py` is fragile and could be cleaner.
**File**: `src/microtorch/tensor/utils.py:23-27`

## 📝 Code Quality Issues

### 8. Unused variable in ReLU class
**Issue**: `activation.py` line 9 declares `inplace: bool` but never initializes or uses it.
**File**: `src/microtorch/nn/modules/activation.py:9`

### 9. Inconsistent parameter storage in Optimizer
**Issue**: Optimizer stores parameters as Iterable but should convert to list for safety.
**File**: `src/microtorch/optim/optimizer.py:20`

### 10. Missing docstrings
**Issue**: Several methods lack proper documentation (e.g., `__init__` methods in modules).
**Files**: Various module files

### 11. Dead code in MNIST example
**Issue**: Example `mnist.py` has commented evaluation code (lines 105-116) that should be removed or fixed.
**File**: `examples/src/microtorch_examples/mnist/mnist.py:105-116`

### 12. Redundant model creation
**Issue**: `mnist.py` creates the model twice (lines 58 and 85).
**File**: `examples/src/microtorch_examples/mnist/mnist.py:58,85`

### 13. Type annotation issues
**Issue**: Several places use `Any` where more specific types could be used.
**Files**: Various files throughout the codebase

### 14. No shape validation in cross_entropy
**Issue**: `cross_entropy` doesn't validate that target contains valid class indices (should be in range [0, C-1]).
**File**: `src/microtorch/tensor/functional.py:448-518`

### 15. Memory inefficiency in Tensor.numpy()
**Issue**: `Tensor.numpy()` always copies data even when not necessary for safety.
**File**: `src/microtorch/tensor/tensor.py:60`

## Recommendations

1. **Immediate fixes needed**: Bugs #1, #2, and #3 are critical and will cause incorrect behavior.
2. **Type safety**: Add more specific type annotations and runtime validations.
3. **Code cleanup**: Remove dead code and unused variables.
4. **Documentation**: Add missing docstrings for better code maintainability.
5. **Testing**: Ensure test coverage for gradient computations and edge cases.