from microtorch.nn.modules.activation import ReLU, Softmax
from microtorch.nn.modules.linear import Linear
from microtorch.nn.modules.module import Module
from microtorch.tensor.tensor import Tensor


def test_get_parameters_with_mnist_model():
    class Model(Module[Tensor]):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(784, 128)
            self.relu = ReLU()
            self.linear2 = Linear(128, 10)
            self.softmax = Softmax(dim=-1)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.softmax(x)
            return x

    model = Model()
    parameters = list(model.named_parameters())
    assert len(parameters) == 4
    assert all(isinstance(param, Tensor) for _, param in parameters)
    assert all(param.requires_grad for _, param in parameters)

    params_dict = {name: param for name, param in parameters}
    assert "linear1.weight" in params_dict
    assert "linear1.bias" in params_dict
    assert "linear2.weight" in params_dict
    assert "linear2.bias" in params_dict
    assert params_dict["linear1.weight"].shape == (784, 128)
    assert params_dict["linear1.bias"].shape == (128,)
    assert params_dict["linear2.weight"].shape == (128, 10)
    assert params_dict["linear2.bias"].shape == (10,)
