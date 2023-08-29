"""Exception classes for Gamspy"""


class GamspyException(Exception):
    """Plain Gamspy exception."""


class EarlyQuit(GamspyException):
    """Raised when there is a KeyboardInterrupt."""


class GamsException(GamspyException):
    """Raised when there is an error during the execution of GAMS"""

    FORMAT = "Gamspy failed to run GAMS program due to %(exc)s."

    def __init__(self, exception: Exception) -> None:
        self.original_exception = exception
        super().__init__(exception)

    def __str__(self) -> str:
        return self.FORMAT % {
            "exc": self.original_exception,
        }


class GdxException(GamspyException):
    """Raised when there is a GDX related error"""

    FORMAT = "Gamspy failed to perform GDX operation due to %(exc)s."

    def __init__(self, exception: Exception) -> None:
        self.original_exception = exception
        super().__init__(exception)

    def __str__(self) -> str:
        return self.FORMAT % {
            "exc": self.original_exception,
        }


class EngineException(GamspyException):
    """Raised when there is a Gams Engine related error"""

    FORMAT = "Gamspy failed to run the job on GAMS Engine due to %(exc)s."

    def __init__(self, exception: Exception) -> None:
        self.original_exception = exception
        super().__init__(exception)

    def __str__(self) -> str:
        return self.FORMAT % {
            "exc": self.original_exception,
        }
