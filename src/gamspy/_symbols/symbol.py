from abc import ABC, abstractmethod


class Symbol(ABC):
    @abstractmethod
    def gamsRepr(self):
        ...

    @abstractmethod
    def getStatement(self):
        ...
