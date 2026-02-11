from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence

    import gamspy as gp

    MatchesType: TypeAlias = dict[
        gp.Equation | Sequence[gp.Equation],
        gp.Variable | Sequence[gp.Variable],
    ]


class FormulationResult:
    """
    FormulationResult class provides a common interface for returning results
    when formulations are called. In the old convention, formulations returned
    a tuple of result variable and list of equations. In some cases it was
    possible to get extra output from the formulation. To provide backwards
    compatibility, FormulationResult class can be unpacked into a result
    variable and list of equations. Also it supports returning extra output in
    unpacking.

    With the FormulationResult you can have more access to underlying symbols
    created such as equations, variables, parameters and sets. Since many
    formulations created symbols with randomized names, it was tedious to find
    intermediate symbols created. FormulationResult has dictionaries where keys
    are expected to be documented in the formulation returning the
    FormulationResult therefore you can access a symbol via its known key.

    For example:

    Examples
    --------
    >>> import gamspy as gp
    >>> m = gp.Container()
    >>> x = gp.Variable(m)
    >>> res = gp.math.activation.relu_with_binary_var(x)
    >>> aux_binary_var = res.variables_created["binary"]

    Therefore, it is important for the formulation returning a
    FormulationResult to properly list the keys to the symbols that are
    created.

    FormulationResult has the following attributes that might be useful:
        - `result`
        - `equations_created`
        - `variables_created`
        - `sets_created`
        - `parameters_created`
        - `matches`
        - `other`
        - `extra_return`

    """

    def __init__(
        self,
        result: gp.Variable | gp.Parameter | None = None,
        equations_created: dict[str, gp.Equation] | None = None,
        extra_return: gp.Variable | MatchesType | None = None,
    ) -> None:
        self.result = result
        self.equations_created: dict[str, gp.Equation] = (
            equations_created if equations_created is not None else {}
        )

        self.variables_created: dict[str, gp.Variable] = {}
        self.sets_created: dict[str, gp.Set] = {}
        self.parameters_created: dict[str, gp.Parameter] = {}
        self.matches: MatchesType = {}
        self.other: dict[str, Any] = {}

        # Extra return is only here for backwards compat reasons
        # try to use other when you can
        self.extra_return = extra_return

    def __len__(self):
        # For backwards compat
        return 2 + (self.extra_return is not None)

    def __iter__(self):
        # For backwards compat
        if self.extra_return is None:
            return [
                self.result,
                list(self.equations_created.values()),
            ].__iter__()

        return [
            self.result,
            self.extra_return,
            list(self.equations_created.values()),
        ].__iter__()

    def __str__(self) -> str:
        return (
            "FormulationResult(\n"
            + "    Equations: "
            + str(self.equations_created.keys())
            + "\n"
            + "    Variables: "
            + str(self.variables_created.keys())
            + "\n"
            + "    Sets: "
            + str(self.sets_created.keys())
            + "\n"
            + "    Parameters: "
            + str(self.parameters_created.keys())
            + "\n"
            + "    Num Matches: "
            + str(len(self.matches))
            + "\n"
            + "    Other: "
            + str(self.other)
            + "\n"
            + ")"
        )
