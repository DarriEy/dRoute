"""
droute - Differentiable River Routing Library

This package is a thin alias for the compiled extension module.
"""

from importlib import import_module
from ._version import __version__

__author__ = "Darri Eythorsson"

try:
    _module = import_module("_droute_core")
except ImportError as exc:
    raise ImportError(
        "droute requires the compiled extension module '_droute_core'. "
        "Please ensure the package is properly installed.\n"
        "Install with: pip install droute\n"
        "Or for development: pip install -e ."
    ) from exc

globals().update(_module.__dict__)

__all__ = getattr(
    _module,
    "__all__",
    [name for name in _module.__dict__ if not name.startswith("_")],
)
