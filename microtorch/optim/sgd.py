from microtorch.nn.modules.parameter import Parameter
from microtorch.optim.optimizer import Optimizer

# pyright: reportPrivateUsage=false


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 0.01,
    ):
        """
        Stochastic Gradient Descent (SGD) optimizer.

        Args:
            parameters (list): List of tensors (numpy arrays) to optimize.
            lr (float): Learning rate.
        """
        super().__init__(parameters, lr)

    def step(self):
        """
        Performs a single optimization step.
        """
        for param in self.parameters:
            if param.grad is None:
                continue  # Skip parameters with no gradient

            param._data -= self.lr * param.grad
