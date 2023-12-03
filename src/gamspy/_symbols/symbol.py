from abc import ABC
from abc import abstractmethod


class Symbol(ABC):
    @abstractmethod
    def gamsRepr(self):
        """Representation of the symbol in GAMS"""

    @abstractmethod
    def getStatement(self):
        """Declaration string of the symbol in GAMS"""
