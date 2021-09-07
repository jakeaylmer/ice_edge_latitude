"""
---------------------------------------------------------
DOCUMENTATION FOR:
Ice-edge latitude module of the diagnostics sub-package
(ice_edge_latitude.py)
---------------------------------------------------------

This module contains functions for computing diagnostics
primarily from sea ice concentration fields. See function
documentation for details:


FUNCTION: ice_edge():
 --> latitude of the sea ice edge for global gridded
     sea ice concentration data.


FUNCTION: get_ice_edge():
 --> latitude of the sea ice edge for single hemisphere,
     ordered data. This is the direct implementation of
     the algorithm; generally use ice_edge() which
     provides a more convenient interface for gridded
     data.


FUNCTION: split_iel_by_region():
 --> Split global ice edge data [e.g., as returned by
     the ice_edge() function] into specific regions,
     e.g., to extract the north Atlantic sea ice edge
     time series.


These functions are designed to work with the CMIP6 sea
ice concentration field (siconc or siconca), but should
work with other sources (e.g. previous phases of CMIP,
or satellite concentration data).

---------------------------------------------------------
"""

import numpy as np
from utilities import utils, regions, math_functions as math



def ice_edge(lat, siconc, threshold=0.15, y=100, poleward=False,
        give_deg_n=True, details=True):
    """Calculate sea ice edge latitudes from gridded global
    sea ice concentration.
    
    This is essentially a wrapper function for the get_ice_edge()
    function (which calculates ice-edge latitudes), that passes
    global gridded data in the required the format. For details
    of the algorithm itself, see the documentation for the
    get_ice_edge() function.
    
    
    Parameters
    ----------
    lat : 1D array-like
        Latitude coordinates in degrees north.
    
    siconc : 2D or 3D array-like
        Gridded sea ice concentration (fraction) field. The
        first dimension is the time axis if 3D, the next two
        dimensions are latitude and longitude (in that order).
        
        Land coordinates must be set to NaN.
    
    
    Optional parameters
    -------------------
    threshold : float, default: 0.15
        sea ice concentration defining the ice edge, in the same
        units as siconc.
    
    y : float, default: 100
        meridional distance in kilometers poleward and equatorward of
        the ice edge within which must not contain land.
    
    poleward : bool, default: False
        in the case of multiple ice edges for a given longitude,
        chooses the one nearest the pole if True, or nearest the
        equator if False.
    
    give_deg_n : bool, default: True
        Whether to return all latitude coordinates in degrees north.
        
    details : bool, default: True
        Print progress to the console (e.g. helpful for high
        resolution, long time series data sets).
    
    
    Returns
    -------
    ie_lat_nh, ie_lat_sh
    
    2D arrays of latitude coordinates for the ice edge in the
    northern (NH) and southern (SH) hemispheres, respectively,
    where the first dimension corresponds to time and the
    second to longitude. If siconc input was 2D, then the
    first dimension is collapsed.
    
    """
    
    # If input data is 2D, put into 3D anyway to allow the same
    # loops below to be used:
    if np.ndim(siconc) == 2:
        siconc = np.array([siconc])
        single_time = True  # flag to convert back to 2D at the end
    else:
        single_time = False
    
    # Split siconc data into northern and southern hemispheres:
    j = np.argmin(abs(lat))  # closest index to equator
    
    # Northern hemisphere (NH):
    lat_nh = lat.copy()[j:]  # in degrees north, 0..90N
    siconc_nh = siconc.copy()[:,j:,:]
    
    # For southern hemisphere (SH), need degrees south
    # and 0..90S, which requires reversing the coordinates:
    lat_sh = abs(lat.copy()[0:j][::-1])
    siconc_sh = siconc.copy()[:,0:j,:][:,::-1,:]
    
    # Set up arrays for ice edge (ie) latitude in each hemisphere,
    # first index is time and second is longitude:
    ie_lat_nh = np.zeros((len(siconc), len(siconc[0,0])))
    ie_lat_sh = np.zeros((len(siconc), len(siconc[0,0])))
    
    # Keyword arguments passed to the get_ice_edge() function:
    ie_args = {'threshold':threshold, 'y':y, 'poleward':poleward}
    
    for t in range(len(siconc)):
        
        if details:
            utils._progress(t+1, len(siconc))
        
        ie_lat_nh[t,:] = get_ice_edge(lat_nh, siconc_nh[t], **ie_args)
        ie_lat_sh[t,:] = get_ice_edge(lat_sh, siconc_sh[t], **ie_args)
    
    if give_deg_n:  # convert SH latitudes back to degN
        ie_lat_sh *= -1
    
    if single_time:  # revert back to input type
        ie_lat_nh = ie_lat_nh[0]
        ie_lat_sh = ie_lat_sh[0]
    
    return ie_lat_nh, ie_lat_sh



def get_ice_edge(lat, siconc, threshold=0.15, y=100,
        poleward=False, r_e=6.371E3):
    """Calculate the ice-edge latitude as a function of longitude from
    sea ice concentration data on a fixed latitude-longitude grid in
    one hemisphere.
    
    *** NOTE *** ---------------------------------------- ***
    Generally the function ice_edge() should be called to
    calculate ice edge, not this one.
    
    This function implements the base algorithm, and requires
    the input data to be in a specific format (see below).
    
    For standard global data with siconc(t, lat, lon), the
    function ice_edge() should be used, which makes sure
    this function gets the data in the required format.
    *** ------------------------------------------------- ***
    
    Unique latitudes are returned per longitude. A point is
    considered the 'ice edge' if it:
        
        (1) is on the threshold concentration
        (2) has a great sea ice concentration poleward
        (3) has a lower sea ice concentration equatorward
        (4) has no land within a meridional distance y kilometers
    
    This algorithm is an implementation of that described by
    Eisenman (2010), with the modification that land checking is
    done on the basis of a meridional distance in
    kilometers (here, y) rather than a number of grid points.
    
    
    Parameters
    ----------
    lat : 1D array-like
        Latitude coordinates (degrees north or degrees south,
        i.e. must be positive and increasing from nearest the
        equator, lat[0], to nearest the pole, lat[-1]).
    
    siconc : 2D array-like
        Sea ice concentration as a function of latitude (first
        dimension) and longitude (second dimension).
        
        This data must be gridded and for one hemisphere only.
        
        Any values which are NaN are treated as land.
    
    
    Optional parameters
    -------------------
    threshold : float, default: 0.15
        sea ice concentration defining the ice edge, in the same
        units as siconc.
    
    y : float, default: 100
        meridional distance in kilometers poleward and equatorward of
        the ice edge within which must not contain land.
    
    poleward : bool, default: False
        in the case of multiple ice edges for a given longitude,
        chooses the one nearest the pole if True, or nearest the
        equator if False.
    
    r_e : float, default: 6.371E3
        The radius of Earth in kilometers.
    
    
    Returns
    -------
    1D array
        Ice-edge latitudes as a function of longitude. If any longitude
        does not contain an ice edge, this is set to NumPy.NaN.
    
    
    References
    ----------
    Eisenman, I., 2010: Geographic muting of changes in the Arctic
    sea ice cover. Geophys. Res. Lett., 37, L16501,
    doi:10.1029/2010GL043741.
    
    """
    
    # ------------------------------------------------------- #
    # Land checking sub-function: returns bool; if True,
    # land checking is passed (i.e. no land within meridional
    # distance y of currently considered ice edge location):
    def _land_check(lat_s, k):
        id = abs(lat-lat_s) < dlat
        return not np.isnan(siconc[id,k]).any()
    # ------------------------------------------------------- #
    
    dlat = (180*y)/(np.pi*r_e)  # convert y to latitude increment
    
    n_lon = len(siconc[0])  # number of longitude coordinates
    n_lat = len(lat)  # number of latitude coordinates
    
    # Set up ice-edge latitude array that will be returned.
    # A unique value is returned for each longitude, if any.
    # If there are no valid points at a given longitude, this
    # is set to NaN at the corresponding index. So start with
    # everything as NaN, then fill in points as they are found:
    ie_lat = np.nan*np.ones(n_lon)
    
    # Don't need to find all possible latitudes at a given longitude:
    # Only need the most poleward one or most equatorward one,
    # depending on the settings. If poleward, start from the pole:
    if poleward:
    
        for k in range(n_lon):  # for each longitude
            
            j = n_lat - 1  # start at pole
            
            # Flag to break the while loop when the ice edge is found:
            searching = True
            
            # Need to keep searching (decreasing j) until either:
            #  (1) ice edge is found (at which point searching is
            #      set to be False, or
            #  (2) there are no more latitudes to check (j == 0):
            while searching and j > 0:
                
                # Check if the current latitude pair (j-1, j)
                # satisfies the siconc threshold criteria:
                if (siconc[j-1,k] < threshold
                    and siconc[j,k] >= threshold):
                    
                    # It does: find the actual ice edge (lat_s)
                    # by interpolating between the coordinate pair:
                    lat_s = math.linear_interpolate(
                        siconc[j-1:j+1,k], lat[j-1:j+1],
                        threshold)
                    
                    # Check whether land exists within a distance
                    # y of lat_s:
                    if _land_check(lat_s, k):
                        
                        # This point passes the land checking, so
                        # becomes the ie_lat for this longitude (k):
                        ie_lat[k] = lat_s
                        
                        # Have now found the most poleward ice edge
                        # for this longitude, so do not need to
                        # continue searching at this longitude:
                        searching = False
                
                # Move on to the next (more equatorward) latitude:
                # (Do this regardless of what the outcome of
                # checking above: we either need to move on, or we
                # found the ice edge and searching was set to False,
                # and this will not matter anyway):
                j -= 1
    
    else:  # want the 'nearest equator' points instead
        
        for k in range(n_lon):  # for each longitude
            
            j = 0  # start at equator
            
            # Flag to break the while loop when the ice edge is found:
            searching = True
            
            # Need to keep searching (increasing j) until either:
            #  (1) ice edge is found (at which point searching is
            #      set to be False, or
            #  (2) there are no more latitudes to check
            #      (j == n_lat - 1):
            while searching and j < n_lat - 1:
                
                # Check if the current latitude pair (j, j+1)
                # satisfies the siconc threshold criteria:
                if (siconc[j,k] < threshold
                    and siconc[j+1,k] >= threshold):
                    
                    # It does: find the actual ice edge (lat_s)
                    # by interpolating between the coordinate pair:
                    lat_s = math.linear_interpolate(
                        siconc[j:j+2,k], lat[j:j+2],
                        threshold)
                    
                    # Check whether land exists within a distance
                    # y of lat_s:
                    if _land_check(lat_s, k):
                        
                        # This point passes the land checking, so
                        # becomes the ie_lat for this longitude (k):
                        ie_lat[k] = lat_s
                        
                        # Have now found the most poleward ice edge
                        # for this longitude, so do not need to
                        # continue searching at this longitude:
                        searching = False
                
                # Move on to the next (more poleward) latitude:
                # (Do this regardless of what the outcome of
                # checking above: we either need to move on, or we
                # found the ice edge and searching was set to False,
                # and this will not matter anyway):
                j += 1
    
    return ie_lat



def split_iel_by_region(ie_lon, ie_lat, region=regions.north_atlantic,
        fill_value=np.nan):
    """Extract sea ice-edge points in a specified region.
    
    Parameters
    ----------
    
    These should be as output from the ice_edge() function:
    
    ie_lon : 1D array-like
        Longitude coordinates of ice-edge points.
    
    ie_lat : 1D or 2D array-like
        Latitude coordinates of ice-edge points (if 2D, first
        dimension is time, second dimension is longitude).
    
    
    Optional parameters
    -------------------
    region : { 'arbitrary key' : [lon_min, lon_max, lat_min, lat_max]}
        where the length-4 array specifies a longitude-latitude box
        defining the region over which to extract ice-edge points.
        Longitude must be specified in degrees east (i.e. 0 to +360),
        and latitude in degrees north (-90 to +90).
        
        Multiple regions may be specified by including multiple
        dictionary entries. For standard cases such as the ocean basins,
        the module 'regions' is provided in the utilities package.
        The default here is the north_atlantic region in that module.
    
    fill_value : float or NaN, default: NaN
        Value for filling ragged list-array into 2D array.
    
    
    Returns
    -------
    ie_lon_reg, ie_lat_reg : arrays
    
    The filtered longitude and latitude points. Missing values are set
    to numpy.NaN (this occurs because in general there are a different
    number of ice edge points (lon_ie, lat_ie) at each time).
    
    """
    
    # If input data is 1D (only one set), put into 2D array
    # anyway to allow the same loops below to be used:
    if np.ndim(ie_lat) == 1:
        ie_lat = np.array([ie_lat])
        single_time = True  # flag to convert back at the end
    else:
        single_time = False
    
    # For checking coincidence with the region domain, set
    # longitudes to 0 <= x < 360, which the regions are defined on
    # (however, data will be saved using the original
    # longitude range):
    ie_lon_s = utils.convert_longitude_range(ie_lon)
    
    ie_lon_reg = []  # ice-edge longitudes for this region
    ie_lat_reg = []  # ice-edge latitudes for this region
    
    for t in range(len(ie_lat)):  # iterate over time
        
        ie_lon_reg_t = []  # longitudes for this time (if any)
        ie_lat_reg_t = []  # latitudes for this time (if any)
        
        for j in range(len(ie_lon)):  # check every lon/lat pair
            
            # Create a flag that will be set True if a
            # valid point (coincides with any sub-region)
            # is found:
            valid = False
            
            # Iterate over sub-regions:
            for a in range(len(region.keys())):
                
                sub = list(region.keys())[a]  # sub-region name
                
                # Check if this lon/lat pair is within the
                # boundary of this region (using its longitude
                # coordinate in the standard 0 to 360 range):
                if (region[sub][0] <= ie_lon_s[j]) and (
                    region[sub][1] >= ie_lon_s[j]) and (
                    region[sub][2] <= ie_lat[t,j]) and (
                    region[sub][3] >= ie_lat[t,j]):
                    
                    valid = True
            
            if valid:
                # Save longitudes from the original range:
                ie_lon_reg_t.append(ie_lon[j])
                ie_lat_reg_t.append(ie_lat[t,j])
        
        # Finished checking all lat/lon pairs in the input set
        # for this time. Now, append the ones valid for this
        # region as an array in the time-series list:
        ie_lon_reg.append(np.array(ie_lon_reg_t))
        ie_lat_reg.append(np.array(ie_lat_reg_t))
    
    # In general, each time step will have a different number of
    # points (e.g. as the ice edge retreats away from land). In other
    # words, we have a 'ragged-right array' (like this comment). To
    # save data in a consistent way, as an array, fill in the gaps:
    ie_lon_reg = utils.fill_list_array(
        ie_lon_reg, fill_value=fill_value)
    ie_lat_reg = utils.fill_list_array(
        ie_lat_reg, fill_value=fill_value)
    
    if single_time:
        # convert back to input format:
        ie_lon_reg = ie_lon_reg[0]
        ie_lat_reg = ie_lat_reg[0]
    
    return ie_lon_reg, ie_lat_reg
