"""
Backwards compatibility shim for pydmc_route -> droute.

This module provides backwards compatibility for code that imports pydmc_route.
The package has been renamed to droute. Please update your imports:

    Old: import pydmc_route
    New: import droute

This compatibility layer will be deprecated in a future version.
"""

import sys
import warnings

# Issue deprecation warning first
warnings.warn(
    "The 'pydmc_route' package name is deprecated. "
    "Please use 'import droute' instead. "
    "This compatibility layer will be removed in version 1.0.0.",
    DeprecationWarning,
    stacklevel=2
)

# Import droute and make everything available
import droute

# Copy all public attributes from droute to this module
__all__ = getattr(droute, '__all__', [name for name in dir(droute) if not name.startswith('_')])

# Make this module act as an alias to droute
for name in __all__:
    globals()[name] = getattr(droute, name)

# Also expose version and author
__version__ = droute.__version__
__author__ = droute.__author__
