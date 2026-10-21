from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._algebra.expression as expression
import gamspy._algebra.operable as operable
import gamspy._symbols as syms
import gamspy.utils as utils
from gamspy._notation import LATEX, UNICODE
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from gamspy import Container
    from gamspy._notation import Notation
    from gamspy._types import OperableType


class ExtrinsicFunction(operable.Operable):
    """
    Extrinsic function registered by the user. Extrinsic functions
    are limited to 20 scalar arguments and return a scalar value.
    """

    def __init__(self, container: Container, name: str) -> None:
        self.container = container
        self.name = name
        self.args: tuple = ()
        self.domain: list = []
        self.dimension = 0

    def __len__(self) -> int:
        if self.records is not None:
            return len(self.records.index)

        return 0

    @property
    def records(self) -> pd.DataFrame | None:
        """
        Evaluates the extrinsic function and returns the resulting records.

        Returns
        -------
        pd.DataFrame | None
        """
        with self.container._temp_symbol(syms.Parameter, self.domain) as temp_param:
            temp_param[...] = self
        return temp_param.records

    def toValue(self) -> float:
        """
        Convenience method to return expression records as a Python float.
        Only possible if there is a single record as a result of the expression evaluation.

        Returns
        -------
        float
        """
        records = self.records
        if records is None:
            raise ValidationError(
                "Could not get the value of the extrinsic function. Please report to support@gams.com."
            )

        return records["value"][0]

    def __eq__(self, other: OperableType) -> expression.Expression:
        return expression.Expression(self, "=e=", other)

    def __hash__(self) -> int:
        return id(self)

    def __ne__(self, other: OperableType) -> expression.Expression:
        return expression.Expression(self, "ne", other)

    def __call__(self, *args: OperableType | str, **kwds: object) -> ExtrinsicFunction:
        if kwds:
            raise ValidationError("External functions do not accept keyword arguments")

        self.args = args
        return self

    def __str__(self) -> str:
        representation = self.name

        if self.args:
            arg_strs = []
            for arg in self.args:
                arg_str = (
                    str(arg) if isinstance(arg, (int, float, str)) else arg.gamsRepr()
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
        return self._math_repr(LATEX)

    def unicodeRepr(self) -> str:
        """
        Representation of this ExtrinsicFunction in mathematical notation with
        Unicode characters.

        Returns
        -------
        str
        """
        return self._math_repr(UNICODE)

    def _math_repr(self, notation: Notation) -> str:
        representation = notation.operator_name(self.name)

        if self.args:
            arg_strs = []
            for arg in self.args:
                arg_str = (
                    notation.escape(arg)
                    if isinstance(arg, str)
                    else expression.get_operand_math_repr(arg, notation)
                )
                arg_strs.append(arg_str)

            args = ",".join(arg_strs)

            representation = f"{representation}({args})"

        return representation


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
    ) -> None:
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
            library_str += f"function {gamspy_name} / {self.name}.{actual_name} /;\n"

        return library_str

    def __getattr__(self, name: str) -> ExtrinsicFunction:
        if name in self.functions:
            return ExtrinsicFunction(self.container, name)

        return self.__getattribute__(name)
