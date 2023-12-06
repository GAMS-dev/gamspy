from __future__ import annotations

from typing import List
from typing import TYPE_CHECKING
from typing import Union

import gamspy as gp
from gamspy.exceptions import GamspyException

if TYPE_CHECKING:
    from gamspy import Set, Alias, Parameter, Variable, Equation


class Symbol:
    def gamsRepr(self):
        """Representation of the symbol in GAMS"""

    def getStatement(self):
        """Declaration string of the symbol in GAMS"""

    def _container_check(
        self: Union[Set, Parameter, Variable, Equation],
        domain: List[Union[str, Set, Alias]],
    ):
        for set in domain:
            if (
                isinstance(set, (gp.Set, gp.Alias))
                and set.container != self.container
            ):
                raise GamspyException(
                    f"`Domain `{set.name}` must be in the same container"
                    f" with `{self.name}`"
                )
