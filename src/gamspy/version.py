# Starting from Python 3.8, importlib has a metadata submodule
from importlib import metadata

__version__ = metadata.version("gamspy")
