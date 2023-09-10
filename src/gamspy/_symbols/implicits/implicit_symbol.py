from abc import ABC
import gamspy._algebra.condition as condition


class ImplicitSymbol(ABC):
    def __init__(self, parent, name, domain) -> None:
        self.parent = parent
        self.ref_container = parent.ref_container
        self.name = name
        self.domain = domain
        self.where = condition.Condition(self)

    def gamsRepr(self):
        """Representation of the implicit symbol in GAMS"""
