try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata  # type: ignore

__version__ = metadata.version("gamspy")
