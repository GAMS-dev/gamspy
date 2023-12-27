from __future__ import annotations

from typing import TYPE_CHECKING

import gams.transfer as gt

import gamspy as gp
import gamspy._symbols.implicits as implicits
from gamspy.exceptions import ValidationError

if TYPE_CHECKING:
    from gamspy import Set, Alias, Parameter, Variable, Equation


class Symbol:
    def gamsRepr(self):
        """Representation of the symbol in GAMS"""

    def getStatement(self):
        """Declaration string of the symbol in GAMS"""

    def _container_check(
        self: Set | Parameter | Variable | Equation,
        domain: list[str | Set | Alias],
    ):
        for set in domain:
            if (
                isinstance(set, (gp.Set, gp.Alias))
                and set.container != self.container
            ):
                raise ValidationError(
                    f"`Domain `{set.name}` must be in the same container"
                    f" with `{self.name}`"
                )

    def _get_domain_str(self):
        set_strs = []
        for set in self.domain:
            if isinstance(set, (gt.Set, gt.Alias, implicits.ImplicitSet)):
                set_strs.append(set.gamsRepr())
            elif isinstance(set, str):
                set_strs.append("*")

        return "(" + ",".join(set_strs) + ")"
