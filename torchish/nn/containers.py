import itertools
from typing import Any, Iterator, Union

import torch as t

from arena.torchish.nn.backprop.backprop import Parameter, Tensor


class Module:
    _modules: dict[str, "Module"]
    _parameters: dict[str, Parameter]

    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def modules(self):
        """Return the direct child modules of this module."""
        return self.__dict__["_modules"].values()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        """
        _parameters = iter(self._parameters.values())

        if recurse:
            for module in self.modules():
                _parameters = itertools.chain(
                    _parameters, module.parameters(recurse=True)
                )

        return _parameters

    def __setattr__(self, key: str, val: Any) -> None:
        """
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call the superclass.
        """
        if isinstance(val, Parameter):
            self._parameters[key] = val
        elif isinstance(val, Module):
            self._modules[key] = val

        super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Union[Parameter, "Module"]:
        """
        If key is in _parameters or _modules, return the corresponding value.
        Otherwise, raise KeyError.
        """

        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]
        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        super().__getattr__(key)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward!")

    def __repr__(self):
        def _indent(s_, numSpaces):
            return re.sub("\n", "\n" + (" " * numSpaces), s_)

        lines = [
            f"({key}): {_indent(repr(module), 2)}"
            for key, module in self._modules.items()
        ]
        return "".join(
            [
                self.__class__.__name__ + "(",
                "\n  " + "\n  ".join(lines) + "\n" if lines else "",
                ")",
            ]
        )


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(str(i), mod)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            x = mod(x)
        return x
