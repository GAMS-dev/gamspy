from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    pass


class ExtrinsicFunction(operable.Operable):
    """Extrinsic function registered by the user."""

    def __init__(self, name: str):
        self.name = name
        self.args: tuple = ()
        self.domain: list = []

    def __eq__(self, other):
        return expression.Expression(self, "=e=", other)

    def __ne__(self, other):
        return expression.Expression(self, "ne", other)

    def __len__(self):
        return len(self.__str__())

    def __call__(self, *args, **kwds) -> ExtrinsicFunction:
        if kwds:
            raise ValidationError(
                "External functions do not accept keyword arguments"
            )

        self.args = args
        return self

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

    def gamsRepr(self) -> str:
        """
        Representation of this ExtrinsicFunction in GAMS language.

        Returns
        -------
        str
        """
        return self.__str__()

    def getDeclaration(self) -> str:
        """
        Declaration of the ExtrinsicFunction in GAMS

        Returns
        -------
        str
        """
        return self.gamsRepr()

    def latexRepr(self) -> str:
        """
        Representation of this ExtrinsicFunction in Latex.

        Returns
        -------
        str
        """
        return self.gamsRepr()


class ExtrinsicLibrary:
    """
    Extrinsic library registered by the user.

    Parameters
    ----------
    path : str
        Path to the shared object that contains the library.
    functions : dict[str, str]
        Functions to be imported from the extrinsic library.
    """

    def __init__(self, path: str, functions: dict[str, str]):
        self.name = "ext" + utils._get_unique_name()
        self.path = path
        self.functions = functions

    def getDeclaration(self) -> str:
        """
        Declaration of the ExtrinsicLibrary in GAMS

        Returns
        -------
        str
        """
        library_str = f"$funclibin {self.name} {self.path}\n"

        for gamspy_name, actual_name in self.functions.items():
            library_str += (
                f"function {gamspy_name} / {self.name}.{actual_name} /;\n"
            )

        return library_str

    def __getattr__(self, name):
        if name in self.functions:
            return ExtrinsicFunction(name)

        return self.__getattribute__(name)
