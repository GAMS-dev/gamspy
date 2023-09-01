from abc import ABC, abstractmethod


class Symbol(ABC):
    @abstractmethod
    def gamsRepr(self):
        """Representation of the symbol in GAMS"""

    @abstractmethod
    def getStatement(self):
        """Declaration string of the symbol in GAMS"""
