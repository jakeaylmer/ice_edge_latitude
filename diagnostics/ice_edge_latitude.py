"""
---------------------------------------------------------
DOCUMENTATION FOR:
Ice-edge latitude module of the diagnostics sub-package
(ice_edge_latitude.py)
---------------------------------------------------------

This module contains functions for computing diagnostics
primarily from sea ice concentration fields. See function
documentation for details:


FUNCTION: _get_max_lat():
 --> a 'private' routine used by the functions below. It
     determines the largest possible latitude of the ice
     edge as a function of longitude (generally the grid
     point(s) nearest the north pole in the northern
     hemisphere, and the Antarctic coastline in the
     southern hemisphere).


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


def _get_max_lat(lat, land_mask):
    """Determine the most poleward latitude where data is not
    land masked, as a function of longitude. This is needed to
    determine where to the put the ice edge on meridians that
    are totally ice free.
    
    In general, this should return the value closest to the
    north pole for northern hemisphere data, and latitudes
    closest to the coastline of Antarctica for southern
    hemisphere data.
    
    
    Parameters
    ----------
    lat : 1D array of shape (n_lat,)
        Latitude coordinates in degrees north or degrees south.
    
    land_mask : 2D array like of shape (n_lat, n_lon)
        Land mask, set to np.nan where land and any other finite
        value otherwise.
    
    
    Returns
    -------
    max_lat : 1D array of shape (n_lon,)
        The largest value in lat where land is not present based
        on land_mask, as a function of longitude.
        
        If all latitudes are land (equal to nan) for a given
        longitude (which can happen from interpolation artefacts
        arising from longitude wrapping, for instance), then
        the value of max_lat for that longitude is set to nan.
    
    """
    
    n_lon = np.shape(land_mask)[1]
    
    max_lat = np.zeros(n_lon).astype(lat.dtype)
    
    for k in range(n_lon):
        if np.isnan(land_mask[:,k]).all():
            max_lat[k] = np.nan
        else:
            max_lat[k] = np.max(lat[~np.isnan(land_mask[:,k])])
    
    return max_lat



def ice_edge(lat, siconc, threshold=0.15, y=100, poleward=False,
        lat_cutoff=(30.0, -30.0),
        set_ice_free_meridians_to_missing=(False, False),
        give_deg_n=True, details=True):
    """Calculate northern and southern hemisphere sea ice edge
    latitudes from gridded global sea ice concentration.
    
    This is a wrapper function for the get_ice_edge() function
    (which calculates ice-edge latitudes), that passes global
    gridded data in the required the format. For details of the
    algorithm itself, see the documentation for the
    get_ice_edge() function.
    
    
    Parameters
    ----------
    lat : 1D array, shape (n_lat,)
        Latitude coordinates in degrees north.
    
    siconc : 2D or 3D array of shape (n_t, n_lat, n_lon)
        Gridded sea ice concentration (fraction) field. The
        first (time) axis may be omitted.
        
        Land coordinates must be set to NaN.
    
    
    Optional parameters
    -------------------
    threshold : float, default: 0.15
        sea ice concentration defining the ice edge, in the
        same units as siconc.
    
    y : float, default: 100
        meridional distance in kilometers poleward and equator-
        ward of the ice edge within which must not contain land.
    
    poleward : bool, default: False
        in the case of multiple ice edges for a given
        longitude, chooses the one nearest the pole if True, or
        nearest the equator if False.
    
    lat_cutoff : tuple of float, default = (30.0, 30.0)
        Minimum latitudes with which to cut input data
        (north, south) in degrees north and degrees south,
        respectively (values are intepreted in those respective
        units anyway, regardless of their input sign which is
        ignored).
    
    set_ice_free_meridians_to_missing : tuple of bool
        Whether to always set ice edge latitudes as missing for
        ice free meridians (if True), or to the maximum ocean
        grid latitude at such longitudes (if False), for the
        northern (index 0) and southern (index 1) hemispheres.
        
        Default: (False, False)
    
    give_deg_n : bool, default: True
        Whether to return all latitude coordinates in degrees
        north (otherwise, southern hemisphere data is in
        degrees south).
    
    details : bool, default: True
        Print progress to the console (e.g., helpful for high-
        resolution, long time series data).
    
    
    Returns
    -------
    ie_lat_nh, ie_lat_sh
    
    2D arrays of latitude coordinates for the ice edge in the
    northern (NH) and southern (SH) hemispheres, respectively,
    where the first dimension corresponds to time and the
    second to longitude. If siconc input was 2D, then there
    is no corresponding time axis (i.e., the arrays are 1D
    containing the ice edge latitude as a function of longitude
    only).
    
    """
    
    # If input data is 2D, put into 3D anyway to allow the same
    # loops below to be used:
    if np.ndim(siconc) == 2:
        siconc = np.array([siconc])
        single_time = True  # flag to convert back to 2D at end
    else:
        single_time = False
    
    n_t, n_lat, n_lon = np.shape(siconc)
    
    # Split siconc data into northern and southern hemispheres:
    j_nh = np.argmin(abs(lat - abs(lat_cutoff[0])))
    j_sh = np.argmin(abs(lat + abs(lat_cutoff[1])))
    
    # Northern hemisphere (NH):
    lat_nh = lat.copy()[j_nh:]  # in degrees north, 0..90N
    siconc_nh = siconc.copy()[:,j_nh:,:]
    
    # For southern hemisphere (SH), need degrees south
    # and 0..90S, which requires reversing the coordinates:
    lat_sh = abs(lat.copy()[0:j_sh+1][::-1])
    siconc_sh = siconc.copy()[:,0:j_sh+1,:][:,::-1,:]
    
    # Set up arrays for ice edge (ie) latitude in each
    # hemisphere, first index is time and second is longitude:
    ie_lat_nh = np.zeros((n_t, n_lon)).astype(lat.dtype)
    ie_lat_sh = np.zeros((n_t, n_lon)).astype(lat.dtype)
    
    # Most-poleward latitudes as a function of longitude,
    # determined from the first siconc time step (assumes
    # the same land mask is present at all time steps).
    # 
    # Unless ice-free meridians are to be set to missing,
    # in which case just set as NaN for each longitude:
    if set_ice_free_meridians_to_missing[0]:
        ice_free_meridian_values_nh = np.nan*np.ones(n_lon)
    else:
        ice_free_meridian_values_nh = \
            _get_max_lat(lat_nh, siconc_nh[0])
    
    # Similarly for southern hemisphere:
    if set_ice_free_meridians_to_missing[1]:
        ice_free_meridian_values_sh = np.nan*np.ones(n_lon)
    else:
        ice_free_meridian_values_sh = \
            _get_max_lat(lat_sh, siconc_sh[0])
    
    # Common keyword arguments passed to get_ice_edge():
    gie_kw = {'threshold':threshold, 'y':y, 'poleward':poleward}
    
    for t in range(n_t):
        
        if details:
            utils._progress(t+1, n_t)
        
        ie_lat_nh[t,:] = get_ice_edge(lat_nh, siconc_nh[t],
            ice_free_meridian_values_nh, **gie_kw)
        ie_lat_sh[t,:] = get_ice_edge(lat_sh, siconc_sh[t],
            ice_free_meridian_values_sh, **gie_kw)
    
    if give_deg_n:  # convert SH latitudes back to degN
        ie_lat_sh *= -1.0
    
    if single_time:
        ie_lat_nh = ie_lat_nh[0]
        ie_lat_sh = ie_lat_sh[0]
    
    return ie_lat_nh, ie_lat_sh



def get_ice_edge(lat, siconc, ice_free_meridian_values,
        threshold=0.15, y=100.0, poleward=False, r_e=6.371E3):
    """Calculate the ice-edge latitude as a function of
    longitude from sea ice concentration data on a fixed
    latitude-longitude grid in one hemisphere.
    
    
    *** NOTE *** ---------------------------------------- ***
    Generally the function ice_edge() should be called to
    calculate ice edge, not this one.
    
    This function implements the base algorithm, and requires
    the input data to be in a specific format (see below).
    
    For standard global data with siconc(t, lat, lon), the
    function ice_edge() should be used, which ensures this
    function gets the data in the required format.
    *** ------------------------------------------------- ***
    
    
    Unique latitudes are returned per longitude. A point
    (lat_s, lon_s) is considered the 'ice edge' if:
        
        (1) siconc(lat_s, lon_s) = threshold,
        (2) siconc >= threshold at the next poleward grid
            point,
        (3) siconc < threshold at the next equatorward grid
            point, and
        (4) has no land within a meridional distance y km
    
    This algorithm is an implementation of that described by
    Eisenman (2010), with the modification that land checking is
    done on the basis of a meridional distance in kilometers
    (here, y) rather than a number of grid points.
    
    
    Parameters
    ----------
    lat : 1D array, shape (n_lat,)
        Latitude coordinates (degrees north or degrees south,
        i.e., must be positive and increasing from nearest the
        equator, lat[0], to nearest the pole, lat[-1]).
    
    siconc : 2D array, shape (n_lat, n_lon)
        Sea ice concentration as a function of latitude (first
        dimension) and longitude (second dimension). This data
        must be gridded and for one hemisphere only. For points
        which are land, the value of siconc must be NaN.
    
    ice_free_meridian_values : 1D array, shape (n_lon,)
        Values to set at each longitude when there is no ice
        edge specifically because there is no sea ice above the
        threshold (i.e., an ice free meridian). This is distinct
        from longitudes that have no ice edge because of land
        constraints (which are set to NaN).
        
        Usually this corresponds to the latitudes nearest the
        north pole for northern hemisphere data, or nearest the
        Antarctic coast for southern hemisphere data. Ice-free
        meridians can also be set to NaN, or any other unique
        flag such as -999.0, if desired, by passing an array
        of NaN or other fixed value such as -999.0 for this
        parameter instead.
    
    
    Optional parameters
    -------------------
    threshold : float, default: 0.15
        Sea ice concentration defining the ice edge, in the
        same units as siconc.
    
    y : float, default: 100.0
        Meridional distance in kilometers poleward and
        equatorward of the ice edge within which must not be
        land (identified by NaN values in siconc).
    
    poleward : bool, default: False
        In the case of multiple ice edges for a given
        longitude, choose the one nearest the pole if True,
        or nearest the equator if False (default).
    
    r_e : float, default: 6.371e3
        The radius of Earth in kilometers.
    
    
    Returns
    -------
    1D array, shape (n_lon,)
        Ice-edge latitudes as a function of longitude. If any
        longitude does not contain an ice edge because of land
        constraints, this is set to NaN. If there is no ice
        edge because the whole meridian is ice free, it is set
        to the corresponding value in most_poleward_lat.
    
    
    References
    ----------
    Eisenman, I., 2010: Geographic muting of changes in the
    Arctic sea ice cover. Geophys. Res. Lett., 37, L16501,
    doi:10.1029/2010GL043741.
    
    """
    
    def _land_check(lat_s, k):
        """Land checking subroutine: returns True if land
        checking is passed, meaning there is no land with a
        meridional distance y of the identified latitude
        lat_s (which need not be a grid point).
        """
        jlat = abs(lat-lat_s) < dlat
        return not np.isnan(siconc[jlat,k]).any()
    
    dlat = (180.0*y)/(np.pi*r_e)  # convert y to lat increment
    
    n_lon = len(siconc[0])  # number of longitude coordinates
    n_lat = len(lat)  # number of latitude coordinates
    
    # Set up ice-edge latitude array that will be returned.
    # A unique value is returned for each longitude, if any.
    # If there are no valid points at a given longitude, this
    # is set to NaN at the corresponding index. Start with
    # everything as NaN, then fill in points as they are found:
    ie_lat = np.nan*np.ones(n_lon).astype(lat.dtype)
    
    # Determine ice-free meridians in advance: where all lats
    # at a given lon are below threshold and not land, i.e.,
    # NaN -- set land locations to zero to pass the condition
    # in where, and apply all condition on axis=0 which is lat:
    ice_free_meridian = np.all(
        np.where(np.isnan(siconc), 0.0, siconc) < threshold,
        axis=0)
    
    # Don't need to find all possible latitudes at a given
    # longitude -- only need the most poleward one or most
    # equatorward one, depending on the settings. If poleward,
    # start from the pole:
    if poleward:
    
        for k in range(n_lon):  # for each longitude
            
            # Set flag "searching" to break the while loop when
            # the ice edge is found. However, only need to do
            # the loop if this is not an ice-free meridian:
            if ice_free_meridian[k]:
                ie_lat[k] = ice_free_meridian_values[k]
                searching = False
            else:
                searching = True
            
            j = n_lat - 1  # start at pole
            
            # Keep searching (decreasing j) until either:
            #  (1) ice edge is found (at which point searching
            #      is set to False), or
            #  (2) there are no more latitudes to check
            #      (j == 0):
            while searching and j > 0:
                
                # Check if the current latitude pair (j-1, j)
                # satisfies the siconc threshold criteria:
                if (siconc[j-1,k] < threshold
                    and siconc[j,k] >= threshold):
                    
                    # It does: find the actual ice edge (lat_s)
                    # by interpolating between the pair:
                    lat_s = math.linear_interpolate(
                        siconc[j-1:j+1,k], lat[j-1:j+1],
                        threshold)
                    
                    # Check whether land exists within a
                    # distance y of lat_s:
                    if _land_check(lat_s, k):
                        
                        # This point passes the land
                        # checking, so becomes the
                        # ie_lat for this longitude (k):
                        ie_lat[k] = lat_s
                        
                        # No need to continue searching
                        # at this longitude:
                        searching = False
                
                # Move on to the next (more equatorward)
                # latitude -- update j regardless of the
                # outcome of checking above, as we either
                # need to move on or we found the ice edge
                # and searching was set to False and this
                # will not matter anyway:
                j -= 1
    
    else:  # want the 'nearest equator' points instead
        
        for k in range(n_lon):  # for each longitude
            
            # Set flag "searching" to break the while loop when
            # the ice edge is found. However, only need to do
            # the loop if this is not an ice-free meridian:
            if ice_free_meridian[k]:
                ie_lat[k] = ice_free_meridian_values[k]
                searching = False
            else:
                searching = True
            
            j = 0  # start at equator
            
            # Keep searching (increasing j) until either:
            #  (1) ice edge is found (at which point searching
            #      is set to False), or
            #  (2) there are no more latitudes to check
            #      (j == n_lat - 1):
            while searching and j < n_lat - 1:
                
                # Check if the current latitude pair (j, j+1)
                # satisfies the siconc threshold criteria:
                if (siconc[j,k] < threshold
                    and siconc[j+1,k] >= threshold):
                    
                    # It does: find the actual ice edge (lat_s)
                    # by interpolating between the pair:
                    lat_s = math.linear_interpolate(
                        siconc[j:j+2,k], lat[j:j+2], threshold)
                    
                    # Check whether land exists within a
                    # distance y of lat_s:
                    if _land_check(lat_s, k):
                        
                        # This point passes the land
                        # checking, so becomes the
                        # ie_lat for this longitude (k):
                        ie_lat[k] = lat_s
                        
                        # No need to continue searching
                        # at this longitude:
                        searching = False
                
                # Move on to the next (more equatorward)
                # latitude -- update j regardless of the
                # outcome of checking above, as we either
                # need to move on or we found the ice edge
                # and searching was set to False and this
                # will not matter anyway:
                j += 1
    
    return ie_lat



def split_iel_by_region(ie_lon, ie_lat,
        region=regions.north_atlantic,
        fill_value=np.nan):
    """Extract sea ice-edge points in a specified region.
    
    
    Parameters
    ----------
    ie_lon : 1D array, shape (n_lon,)
        Longitude coordinates of ice-edge points.
    
    ie_lat : 1D or 2D array, shape (n_t, n_lon)
        Latitude coordinates of ice-edge points (the first axis
        is time, which may be omitted, and the second is
        longitude).
    
    
    Optional parameters
    -------------------
    region : { 'key' : [lon_min, lon_max, lat_min, lat_max]}
        Each length-4 array, list, or tuple, specifies a
        longitude-latitude box defining the region over which
        to extract ice-edge points. Longitude must be specified
        in degrees east (0 to 360), and latitude in degrees
        north (-90 to +90).
        
        Multiple regions may be specified by including multiple
        dictionary entries. For standard cases such as the
        ocean basins, the module 'regions' is provided in the
        utilities package. The default here is the
        north_atlantic region in that module.
    
    fill_value : float or NaN, default: NaN
        Value for filling ragged list-array into 2D array.
    
    
    Returns
    -------
    ie_lon_reg, ie_lat_reg : arrays
    
    The filtered longitude and latitude points. Missing values
    are set to NaN [this occurs because in general there are a
    different number of ice edge points (lon_ie, lat_ie) at
    each time].
    
    """
    
    # If input data is 1D (only one set), put into 2D array
    # anyway to allow the same loops below to be used:
    if np.ndim(ie_lat) == 1:
        ie_lat = np.array([ie_lat])
        single_time = True  # flag to convert back at the end
    else:
        single_time = False
    
    n_t = len(ie_lat)
    n_lon = len(ie_lon)
    
    # For checking coincidence with the region domain, set
    # longitudes to 0 <= x < 360, which the regions are defined
    # on (however, data will be saved using the original
    # longitude range):
    ie_lon_s = utils.convert_longitude_range(ie_lon)
    
    ie_lon_reg = []  # ice-edge longitudes for this region
    ie_lat_reg = []  # ice-edge latitudes for this region
    
    for t in range(n_t):  # iterate over time
        
        ie_lon_reg_t = []  # longitudes for this time (if any)
        ie_lat_reg_t = []  # latitudes for this time (if any)
        
        for k in range(n_lon):  # check every lon/lat pair
            
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
                if (region[sub][0] <= ie_lon_s[k]) and (
                    region[sub][1] >= ie_lon_s[k]) and (
                    region[sub][2] <= ie_lat[t,k]) and (
                    region[sub][3] >= ie_lat[t,k]):
                    
                    valid = True
            
            if valid:
                # Save longitudes from the original range:
                ie_lon_reg_t.append(ie_lon[k])
                ie_lat_reg_t.append(ie_lat[t,k])
        
        # Finished checking all lat/lon pairs in the input set
        # for this time. Now, append the ones valid for this
        # region as an array in the time-series list:
        ie_lon_reg.append(np.array(ie_lon_reg_t))
        ie_lat_reg.append(np.array(ie_lat_reg_t))
    
    # In general, each time step will have a different number of
    # points (e.g., as the ice edge retreats away from land).
    # In other words, we have a 'ragged-right array' (like this
    # comment). To save data in a consistent way, as an array,
    # fill in the gaps:
    ie_lon_reg = utils.fill_list_array(
        ie_lon_reg, fill_value=fill_value)
    ie_lat_reg = utils.fill_list_array(
        ie_lat_reg, fill_value=fill_value)
    
    if single_time:  # convert back to input format
        ie_lon_reg = ie_lon_reg[0]
        ie_lat_reg = ie_lat_reg[0]
    
    return ie_lon_reg, ie_lat_reg
