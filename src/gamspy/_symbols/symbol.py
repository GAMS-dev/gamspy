from __future__ import annotations

from typing import TYPE_CHECKING, Union

import gams.transfer as gt

import gamspy._symbols.implicits as implicits

if TYPE_CHECKING:
    from gamspy import Alias, Equation, Parameter, Set, Variable

    SymbolType = Union[Alias, Set, Parameter, Variable, Equation]


class Symbol:
    def gamsRepr(self: SymbolType):
        """Representation of the symbol in GAMS"""

    def getStatement(self: SymbolType):
        """Declaration string of the symbol in GAMS"""

    def _get_domain_str(self: SymbolType):
        if isinstance(self.domain_forwarding, bool):
            forwarding = [self.domain_forwarding] * self.dimension
        else:
            forwarding = self.domain_forwarding

        set_strs = []
        for elem, is_forwarding in zip(self.domain, forwarding):
            if isinstance(elem, (gt.Set, gt.Alias, implicits.ImplicitSet)):
                set_str = elem.gamsRepr()
                set_strs.append(set_str + "<" if is_forwarding else set_str)
            elif isinstance(elem, str):
                set_strs.append("*")

        return "(" + ",".join(set_strs) + ")"

    def _mark_forwarded_domain_sets(self):
        for elem in self.domain:
            if hasattr(elem, "modified"):
                elem.modified = False
                elem._is_dirty = True
