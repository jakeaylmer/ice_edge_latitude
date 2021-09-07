"""
---------------------------------------------------------
DOCUMENTATION FOR:
Utilities module of the utilities sub-package
(utils.py)
---------------------------------------------------------

This module contains miscellaneous functions used by
the diagnostics sub-package.

---------------------------------------------------------
"""

import numpy as np



def _progress(i, n, name="... progress"):
    """Print progress to the console for looping routines.
    
    
    Parameters
    ----------
    i : int, > 0
        current iteration.
    
    n : int, > 0
        total number of iterations.
    
    name : str, optional (default: "... progress")
        Label (e.g., of diagnostic being calculated).
    
    """
    
    if i <= 1:
        
        # Make sure to start on a new line:
        print("", end="")
    
    else:
        
        # Use carriage return (\r) and do not start
        # a new line (end="") to overwrite each
        # line, and compute percentage complete:
        print("\r%s: %.0f%%"
            % (name, 100*(i-1)/n), end="")
        
        if i == n:
            # Ensure this always ends up at 100%
            # (Yes, this is cheating a bit, but in most
            # cases it won't be noticeable that it says
            # 100% slightly before actually completing,
            # and it saves having a separate print
            # function at the end of each iteration):
            print("\r%s: 100%%" % name, end="\n")



def fill_list_array(x, fill_value=np.nan):
    """Convert a list of 1D NumPy arrays of differing
    lengths into a 2D NumPy array, filling the spaces
    with a specified constant.
    
    Parameters
    ----------
    x : list of 1D NumPy arrays of arbitrary lengths.
    
    Optional parameters
    -------------------
    fill_value : float or NaN, default: NaN
        Value with which to fill.
    
    Examples
    --------
    
    >>> a = np.array([1, 2, 3, 4])
    >>> b = np.array([1, 2, 3, 4, 5, 6])
    >>> c = np.array([1, 2])
    >>> x = [a, b, c]
    >>> x
    [array([1, 2, 3, 4]), array([1, 2, 3, 4, 5, 6]), array([1, 2])]
    >>>
    >>> fill_list_array(x, fill_value=np.nan)
    array([[ 1.,  2.,  3.,  4., nan, nan],
           [ 1.,  2.,  3.,  4.,  5.,  6.],
           [ 1.,  2., nan, nan, nan, nan]])
    
    """
    
    # Find the maximum length of all input arrays:
    len_max = max([len(xj) for xj in x])
    
    # Set up the filled array of shape
    # ( number of input arrays, max. len of all input arrays )
    # Initially set all values to fill_value:
    x_filled = fill_value*np.ones( (len(x), len_max) )
    
    # Iterate over each input array and populate
    # x_filled with its data:
    for j in range(len(x)):
        x_filled[j,:len(x[j])] = x[j]
    
    return x_filled



def validate_coordinates(x, degrees=False):
    """Check whether coordinates are in radians
    and, if not, convert to radians.
    
    Parameters
    ----------
    x : array_like
        Coordinates to be checked. Values are assumed to
        be in the same coordinates and are not checked
        individually.
    
    degrees : bool, default: False
        Instead check whether coordinates are in degrees,
        converting if required.
    
    Returns
    -------
    xr : same as input type
        Coordinates in radians (or degrees if specified
        with the degrees parameter).
    
    """
    
    if (x > 2*np.pi).any():  # must be in degrees
        xr = x.copy() if degrees else np.pi*x/180
    else:
        xr = 180*x/np.pi if degrees else x.copy()
    
    return xr



def convert_longitude_range(lon, range=360):
    """Put longitude coordinates into the desired range.
    Note that it does not shift data, only values are converted.
    
    Parameters
    ---------
    lon : array-like
        Longitude coordinates in degrees.
    
    
    Optional parameters
    -------------------
    range : int, default: 360
        Specifies the desired range via the maximum positive
        longitude, e.g. 360 implies a range of 0 <= x < 360,
        180 implies a range of -180 <= x < 180, etc.
    
    Returns
    -------
    array-like, new array with shifted longitudes.
    
    """
    
    # Ensure coordinates are in degrees:
    lon_d = validate_coordinates(lon.copy(), degrees=True)
    
    # Shift values above the upper longitude limit:
    lon_new = np.where(lon_d >= range, lon_d - 360, lon_d)
    
    # Shift values below the lower longitude limit (range - 360):
    lon_new = np.where(lon_new < range - 360, lon_new + 360, lon_new)
    
    return lon_new
