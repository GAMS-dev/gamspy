try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

try:
    __version__ = metadata.version("gamspy")
except KeyError:
    # Working with non-compiled version
    __version__ = "0.9.0"
