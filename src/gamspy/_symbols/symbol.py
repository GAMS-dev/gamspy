from __future__ import annotations

import gams.transfer as gt

import gamspy._symbols.implicits as implicits


class Symbol:
    def gamsRepr(self):
        """Representation of the symbol in GAMS"""

    def getStatement(self):
        """Declaration string of the symbol in GAMS"""

    def _get_domain_str(self):
        set_strs = []
        for set in self.domain:
            if isinstance(set, (gt.Set, gt.Alias, implicits.ImplicitSet)):
                set_strs.append(set.gamsRepr())
            elif isinstance(set, str):
                set_strs.append("*")

        return "(" + ",".join(set_strs) + ")"
