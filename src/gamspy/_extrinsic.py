from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy._algebra.expression import Expression
    from gamspy._container import Container


class ExtinsicFunction(operable.Operable):
    def __init__(self, name: str):
        self.name = name
        self.args: tuple | None = None

    def __len__(self):
        return len(self.__str__())

    def __call__(self, *args, **kwds) -> Expression:
        if kwds:
            raise ValidationError(
                "External functions do not accept keyword arguments"
            )

        self.args = args
        return expression.Expression(None, self, None)

    def __str__(self) -> str:
        representation = self.name

        if self.args:
            arg_strs = []
            for arg in self.args:
                arg_str = (
                    str(arg)
                    if isinstance(arg, (int, float, str))
                    else arg.gamsRepr()
                )
                arg_strs.append(arg_str)

            args = ",".join(arg_strs)

            representation = f"{representation}({args})"

        return representation

    def gamsRepr(self):
        return self.__str__()

    def getStatement(self):
        return self.gamsRepr()


class ExtrinsicLibrary:
    def __init__(
        self, container: Container, path: str, name: str, functions: dict
    ):
        self.container = container
        self.path = path
        self.name = name
        self.functions = functions

    def getStatement(self):
        library_str = f"$funclibin {self.name} {self.path}\n"

        for gamspy_name, actual_name in self.functions.items():
            library_str += (
                f"function {gamspy_name} / {self.name}.{actual_name} /;\n"
            )

        return library_str

    def __getattr__(self, name):
        if name in self.functions:
            return ExtinsicFunction(name)

        return self.__getattribute__(name)
