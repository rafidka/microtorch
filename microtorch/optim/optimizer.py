from abc import ABC, abstractmethod

from microtorch.nn.modules.parameter import Parameter


class Optimizer(ABC):
    """
    Base class for all optimizers.
    """

    def __init__(self, parameters: list[Parameter], lr: float):
        """
        Base class for all optimizers.

        Args:
            parameters (list): List of tensors to optimize.
            lr (float): Learning rate.
        """
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        """Performs a single optimization step (must be implemented in subclass)."""
        raise NotImplementedError

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.parameters:
            param.zero_grad()
