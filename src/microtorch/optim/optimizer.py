from abc import ABC, abstractmethod
from collections.abc import Iterable

from microtorch.nn.modules.parameter import Parameter


class Optimizer(ABC):
    """
    Base class for all optimizers.
    """

    def __init__(self, parameters: Iterable[Parameter], lr: float) -> None:
        """
        Base class for all optimizers.

        Args:
            parameters (list): List of tensors to optimize.
            lr (float): Learning rate.
        """
        # Convert to list to avoid iterator exhaustion on multiple iterations
        self.parameters: list[Parameter] = list(parameters)
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        """Performs a single optimization step (must be implemented in subclass)."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Clears the gradients of all optimized parameters."""
        for param in self.parameters:
            param.zero_grad()
