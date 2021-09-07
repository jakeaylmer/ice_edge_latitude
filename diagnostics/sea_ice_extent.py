"""
---------------------------------------------------------
DOCUMENTATION FOR:
Sea ice extent module of the diagnostics sub-package
(sea_ice_extent.py)
---------------------------------------------------------

This module provides a function for computing sea ice
cover diagnostics from sea ice concentration fields.
See function documentation for details:


FUNCTION: sea_ice_extent():
 --> sea ice extent/area between specified contours,
     e.g. area between 15% and 100% contours, without
     concentration weighting (standard sea ice extent)
     or with (standard sea ice area).


These functions are designed to work with the CMIP6 sea
ice concentration field (siconc or siconca), but should
work with other sources (e.g., previous phases of CMIP,
or satellite concentration data).

---------------------------------------------------------
"""

import numpy as np
from utilities import utils



def sea_ice_extent(lat, area, siconc, threshold=[0.15, 1],
        as_area=False, details=True):
    """Calculate sea ice extent-like quantities from sea ice
    concentration data.
    
    Sea ice extent is the sum of all cell areas where the
    sea ice concentration exceeds a minimum threshold (usually
    0.15). This is the default calculation.
    
    This function can also calculate sea ice area, which weights
    the sum by the sea ice concentration, and can account for an
    upper threshold (e.g., to compute the sea ice extent of the
    marginal ice zone with an upper threshold of 0.80).
    
    
    Parameters
    ----------
    lat : 1D or 2D array-like
        Latitude coordinates (on the grid of siconc). If 1D,
        assumes siconc is gridded and this is the corresponding
        latitude axis.
    
    area : 2D array-like
        Grid cell areas (on the grid of siconc).
    
    siconc : 2D or 3D array-like
        Sea ice concentration (fraction) field. The first
        dimension is the time axis if 3D, the other dimensions
        are the grid coordinates which must match lat and area,
        and can have arbitrary geometry (e.g., native ocean grid).
        
        NaN values are ignored.
    
    
    Optional parameters
    -------------------
    threshold : [min., max.], default: [0.15, 1]
        Minimum and maximum threshold concentrations contributing
        to the area calculation. Default corresponds to standard
        definition of sea ice extent.
    
    as_area : bool, default: False
        Calculate sea ice area instead of extent (the former weights
        by sea ice concentration).
    
    details : bool, default: True
        Print progress to the console (e.g., helpful for high
        resolution or long time series data sets).
    
    
    Returns
    -------
    ex_nh : float or array of length of 0-axis of siconc if input 3D
        The northern hemisphere sea ice extent/area, single value or
        time series.
    
    ex_sh : as above but for the southern hemisphere.
    
    """
    
    if np.ndim(lat) == 1:
        # Assume this is the latitude axis of a gridded data set
        # and create a mesh grid:
        _, lat = np.meshgrid(np.ones(len(area[0,:])), lat)
    
    if details:
        diagnostic = 'sea ice ' + ('area' if as_area else 'extent')
        print("Calculating %s, " % (diagnostic)
            + "concentration interval: [%.3f, %.3f]"
            % (threshold[0], threshold[1]))
    
    if np.ndim(siconc) == 2:
        # Add a length-1 axis for the calculation:
        siconc = np.array([siconc])
        single_value = True  # flag to convert back to float
    else:
        single_value = False
    
    # Set up arrays to contain extent/area data for each hemisphere:
    ex_nh = np.zeros(len(siconc))
    ex_sh = np.zeros(len(siconc))
    
    area_nh = np.where(lat > 0, area, 0)
    area_sh = np.where(lat < 0, area, 0)
    
    # First, convert NaN flags (assumed to be land or missing data)
    # to zero (since we sum, these will not contribute):
    siconc = np.where(np.isnan(siconc), 0, siconc)
    
    # Next, eliminate siconcs less than the lower threshold
    # by setting them also to zero:
    siconc = np.where(siconc < threshold[0], 0, siconc)
    
    # Repeat for the upper threshold:
    siconc = np.where(siconc > threshold[1], 0, siconc)
    
    # If we want extent, might as well set set remaining
    # non-zero siconcs to 1. Then, whether we want extent
    # or area, we can just multiply by siconc in the sum:
    if not as_area:
        siconc = np.where(siconc == 0, 0, 1)
    
    for t in range(len(siconc)):  # for each time
        
        if details:  # print progress to console
            utils._progress(t+1, len(siconc))
        
        ex_nh[t] = np.sum(area_nh*siconc[t,:,:])
        ex_sh[t] = np.sum(area_sh*siconc[t,:,:])
    
    if single_value:
        ex_nh = ex_nh[0]
        ex_sh = ex_sh[0]
    
    return ex_nh, ex_sh
