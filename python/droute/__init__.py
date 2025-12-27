"""
droute - Differentiable River Routing Library

This package is a thin alias for the compiled extension module.
"""

from importlib import import_module


try:
    _module = import_module("pydmc_route")
except ImportError as exc:
    raise ImportError(
        "droute requires the compiled extension module. "
        "Build with: pip install -e ."
    ) from exc

globals().update(_module.__dict__)

__all__ = getattr(
    _module,
    "__all__",
    [name for name in _module.__dict__ if not name.startswith("_")],
)
__version__ = getattr(_module, "__version__", "0.5.0")
__author__ = getattr(_module, "__author__", "dMC-Route Authors")
