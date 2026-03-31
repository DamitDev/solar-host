"""Solar Host - Process manager for model inference backends."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("solar-host")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"
