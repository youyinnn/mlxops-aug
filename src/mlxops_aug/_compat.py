"""
Compatibility patches for third-party packages that rely on deprecated
NumPy type aliases removed in NumPy 1.24+.

Import this module before any package that uses the removed aliases
(e.g. gco-wrapper / pygco).

Aliases patched:
  np.bool    -> bool
  np.int     -> np.int_
  np.float   -> np.float64
  np.complex -> np.complex128
  np.object  -> object
  np.str     -> str
  np.float128 -> np.longdouble  (non-Windows only)
"""

import sys
import numpy as np


def patch_numpy_deprecated_aliases() -> None:
    """Restore NumPy type aliases that were removed in NumPy 1.24."""
    _aliases = {
        "bool": bool,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
    }
    for name, dtype in _aliases.items():
        if not hasattr(np, name):
            setattr(np, name, dtype)

    if sys.platform != "win32" and not hasattr(np, "float128"):
        setattr(np, "float128", np.longdouble)


patch_numpy_deprecated_aliases()
