from __future__ import annotations

from typing import TYPE_CHECKING

import gamspy._symbols.alias as alias
import gamspy._symbols.implicits as implicits
import gamspy._symbols.set as gams_set
from gamspy._symbols.implicits.implicit_symbol import ImplicitSymbol

if TYPE_CHECKING:
    from gamspy import Equation, Set


class ImplicitEquation(ImplicitSymbol):
    def __init__(
        self,
        parent: Equation,
        name: str,
        type: str,
        domain: list[Set | str],
    ) -> None:
        """Implicit Equation

        Parameters
        ----------
        parent : Equation
        name : str
        domain : List[Set | str]
        """
        super().__init__(parent, name, domain)
        self.type = type

    def gamsRepr(self) -> str:
        representation = f"{self.name}"
        if len(self.domain):
            set_strs = []
            for set in self.domain:
                if isinstance(
                    set, (gams_set.Set, alias.Alias, implicits.ImplicitSet)
                ):
                    set_strs.append(set.gamsRepr())
                elif isinstance(set, str):
                    if set == "*":
                        set_strs.append(set)
                    else:
                        set_strs.append('"' + set + '"')

            domain_str = "(" + ",".join(set_strs) + ")"

            representation += domain_str

        return representation
