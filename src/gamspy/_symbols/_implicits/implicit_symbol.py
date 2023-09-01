from abc import ABC, abstractmethod


class ImplicitSymbol(ABC):
    @abstractmethod
    def gamsRepr(self):
        """Representation of the implicit symbol in GAMS"""
