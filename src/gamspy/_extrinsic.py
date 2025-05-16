from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols as syms
import gamspy.utils as utils
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import Container


class ExtrinsicFunction(operable.Operable):
    """
    Extrinsic function registered by the user. Extrinsic functions
    are limited to 20 scalar arguments and return a scalar value.
    """

    def __init__(self, container: Container, name: str):
        self.container = container
        self.name = name
        self.args: tuple = ()
        self.domain: list = []
        self.dimension = 0

    @property
    def records(self) -> pd.DataFrame | None:
        """
        Evaluates the extrinsic function and returns the resulting records.

        Returns
        -------
        pd.DataFrame | None
        """
        assert self.container is not None
        temp_name = "a" + utils._get_unique_name()
        temp_param = syms.Parameter._constructor_bypass(
            self.container, temp_name, self.domain
        )
        temp_param[...] = self
        del self.container.data[temp_name]
        return temp_param.records

    def toValue(self) -> float | None:
        """
        Convenience method to return expression records as a Python float. Only possible if there is a single record as a result of the expression evaluation.

        Returns
        -------
        float | None
        """
        records = self.records
        if records is not None:
            return records["value"][0]

        return records

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
    container : str
        Container that the extrinsic library was added to.
    path : str
        Path to the shared object that contains the library.
    functions : dict[str, str]
        Functions to be imported from the extrinsic library.
    """

    def __init__(
        self, container: Container, path: str, functions: dict[str, str]
    ):
        self.container = container
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
            return ExtrinsicFunction(self.container, name)

        return self.__getattribute__(name)
