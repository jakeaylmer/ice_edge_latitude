"""
---------------------------------------------------------
DOCUMENTATION FOR:
Math functions module of utilities sub-package
(math_functions.py)
---------------------------------------------------------

This module contains some general mathematical
procedures used by some of the diagnostic modules.

---------------------------------------------------------
"""

import numpy as np


def linear_interpolate(x, f, x_want=0):
    """Simple linear interpolation (or extrapolation) of f(x).
    
    Parameters
    ---------
    x : array-like of length 2
        Coordinates at which f is known.
    
    f : array-like of length 2
        Values of f at coordinates x.
    
    
    Optional parameters
    -------------------
    x_want : float
        The value of x where f is desired.
    
    
    Returns
    -------
    float, estimated value of f(x_want).
    
    """
    return f[0] + (f[1]-f[0])*(x_want-x[0])/(x[1]-x[0])
