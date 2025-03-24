from collections.abc import Iterator

from microtorch.nn.modules import module
from microtorch.nn.modules.parameter import Parameter

# pyright: reportPrivateUsage=false


class ParameterIterator[T]:
    def __init__(
        self,
        module: "module.Module[T]",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        self.module = module
        self.recurse = recurse
        self.remove_duplicate = remove_duplicate

    def __iter__(self) -> Iterator["Parameter"]:
        seen: set[int] | None = set() if self.remove_duplicate else None
        yield from self._collect_parameters(self.module, seen)

    def _collect_parameters(
        self, module: "module.Module[T]", seen: set[int] | None
    ) -> Iterator["Parameter"]:
        for param in module._parameters.values():
            if seen is None or id(param) not in seen:
                if seen is not None:
                    seen.add(id(param))
                yield param
        if self.recurse:
            for child in module._modules.values():
                yield from self._collect_parameters(child, seen)


class NamedParameterIterator[T]:
    def __init__(
        self,
        module: "module.Module[T]",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        self.module = module
        self.recurse = recurse
        self.remove_duplicate = remove_duplicate

    def __iter__(self) -> Iterator[tuple[str, "Parameter"]]:
        seen: set[int] | None = set() if self.remove_duplicate else None
        yield from self._collect_named_parameters(self.module, prefix="", seen=seen)

    def _collect_named_parameters(
        self, module: "module.Module[T]", prefix: str, seen: set[int] | None
    ) -> Iterator[tuple[str, "Parameter"]]:
        for name, param in module._parameters.items():
            if seen is None or id(param) not in seen:
                if seen is not None:
                    seen.add(id(param))
                yield prefix + name, param
        if self.recurse:
            for child_name, child in module._modules.items():
                child_prefix = prefix + child_name + "."
                yield from self._collect_named_parameters(child, child_prefix, seen)
