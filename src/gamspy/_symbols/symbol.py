from __future__ import annotations
from typing import Union, TYPE_CHECKING

import gams.transfer as gt

import gamspy._symbols.implicits as implicits

if TYPE_CHECKING:
    from gamspy import Alias, Set, Parameter, Variable, Equation

    SymbolType = Union[Alias, Set, Parameter, Variable, Equation]


class Symbol:
    def gamsRepr(self: SymbolType):
        """Representation of the symbol in GAMS"""

    def getStatement(self: SymbolType):
        """Declaration string of the symbol in GAMS"""

    def _get_domain_str(self: SymbolType):
        set_strs = []
        for set in self.domain:  # pylint: disable=E1101
            if isinstance(set, (gt.Set, gt.Alias, implicits.ImplicitSet)):
                set_strs.append(set.gamsRepr())
            elif isinstance(set, str):
                set_strs.append("*")

        return "(" + ",".join(set_strs) + ")"
