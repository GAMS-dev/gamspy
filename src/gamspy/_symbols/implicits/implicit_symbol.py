from __future__ import annotations

from abc import ABC, abstractmethod

import gams.transfer as gt

import gamspy._algebra.condition as condition


class ImplicitSymbol(ABC):
    def __init__(self, parent, name, domain) -> None:
        self.parent = parent
        self.container = parent.container
        self.name = name
        self.domain = domain
        self.where = condition.Condition(self)

    @abstractmethod
    def gamsRepr(self):
        """Representation of the implicit symbol in GAMS"""

    def latexRepr(self):
        from .implicit_set import ImplicitSet

        name = self.name.replace("_", "\_")
        representation = f"\\text{{{name}}}"

        if self.domain:
            set_strs = []
            for elem in self.domain:
                if isinstance(elem, (gt.Set, gt.Alias, ImplicitSet)):
                    set_strs.append(elem.latexRepr())
                elif isinstance(elem, str):
                    set_strs.append("*")

            domain_str = ",".join(set_strs)
            representation = f"{representation}_{{{domain_str}}}"

        return representation
