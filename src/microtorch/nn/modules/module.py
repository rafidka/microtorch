# Python imports
from collections.abc import Iterable
from typing import Any

# Local imports
from microtorch.nn.modules.parameter import Parameter


class Module[T]:
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        self._modules: dict[str, Module[Any]] = {}
        self._parameters: dict[str, Parameter] = {}

    def add_module(self, name: str, module: "Module[T]"):
        self._modules[name] = module

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Module):
            self.add_module(name, value)  # type: ignore
        elif isinstance(value, Parameter):
            self.register_parameter(name, value)
        super().__setattr__(name, value)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> T:
        raise NotImplementedError

    def register_parameter(self, name: str, param: Parameter):
        # TODO Disabling this for now because it is not possible to construct a
        # non-leaf tensor. Consider re-enabling this check in the future.
        # if param._backward is not None:  # type: ignore
        #     raise ValueError(f"Cannot assign non-leaf Tensor as parameter '{name}'.")
        self._parameters[name] = param

    def named_parameters(
        self,
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, Parameter]]:
        # memo: set[Parameter] = set()

        for name, param in self._parameters.items():
            # TODO - Re-enable this in the future and add tests for it.
            # if remove_duplicate:
            #     if param in memo:
            #         continue
            #     memo.add(param)
            yield name, param
        if recurse:
            for name, module in self._modules.items():
                for param_name, param in module.named_parameters(
                    recurse, remove_duplicate
                ):
                    # TODO - Re-enable this in the future and add tests for it.
                    # if remove_duplicate:
                    #     if param in memo:
                    #         continue
                    #     memo.add(param)
                    yield f"{name}.{param_name}", param

    def parameters(
        self,
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterable[Parameter]:
        for _, param in self.named_parameters(recurse, remove_duplicate):
            yield param
