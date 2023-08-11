from enum import Enum


class Problem(Enum):
    LP = "LP"
    NLP = "NLP"
    QCP = "QCP"
    DNLP = "DNLP"
    MIP = "MIP"
    RMIP = "RMIP"
    MINLP = "MINLP"
    RMINLP = "RMINLP"
    MIQCP = "MIQCP"
    RMIQCP = "RMIQCP"
    MCP = "MCP"
    CNS = "CNS"
    MPEC = "MPEC"
    RMPEC = "RMPEC"
    EMP = "EMP"
    MPSGE = "MPSGE"

    @classmethod
    def values(cls):
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class Sense(Enum):
    MIN = "MIN"
    MAX = "MAX"

    @classmethod
    def values(cls):
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class VariableType(Enum):
    BINARY = "BINARY"
    INTEGER = "INTEGER"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    FREE = "FREE"
    SOS1 = "SOS1"
    SOS2 = "SOS2"
    SEMICONT = "SEMICONT"
    SEMIINT = "SEMIINT"

    @classmethod
    def values(cls):
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value


class EquationType(Enum):
    EQ = "EQ"
    GEQ = "GEQ"
    LEQ = "LEQ"
    NONBINDING = "NONBINDING"
    EXTERNAL = "EXTERNAL"
    CONE = "CONE"
    BOOLEAN = "BOOLEAN"

    @classmethod
    def values(cls):
        return list(cls._value2member_map_.keys())

    def __str__(self) -> str:
        return self.value
