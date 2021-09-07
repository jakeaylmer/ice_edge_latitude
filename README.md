# ice_edge_latitude
Python code to calculate the latitude of the sea ice edge from climate model/observation data based on the algorithm of Eisenman (2010). This package also includes a routine for calculating sea ice extent and area, and the code should also work with other climate data provided they are transformed into the required structure.

## Requirements

* Python 3 (tested on 3.7.5)
* NumPy (tested on 1.19.3)
* Tested on Windows and UNIX but should be OS-independent

This package does not provide functions for dealing with raw data, and expects input in the form of NumPy arrays. CMIP data is available in NetCDF format, which can be opened and converted to arrays using the NetCDF4 package. Interpolation of fields onto regular grids can be done on arrays directly with SciPy. The cf-python package is also recommended for interpolation, as it is specifically designed for climate data.

## Package structure

* **ice_edge_latitude** (main package)
* **diagnostics** (sea ice diagnostics sub-package)
    * `ice_edge_latitude.py`
    * `sea_ice_extent.py`
* **utilities** (general functions used by the diagnostics sub-package)
    * `math_functions.py`
    * `regions.py`
    * `utils.py`

## Sea ice diagnostics package: overview

The module `sea_ice_extent.py` provides the following function (see documentation for further details and usage):

* `sea_ice_extent()` calculates sea ice extent or area, by hemisphere and between specified concentration contours, *from sea ice concentration on arbitrary grids*.

The module `ice_edge_latitude.py` provides the following functions:

* `get_ice_edge()` calculates the latitude of the sea ice edge as a function of time and longitude *from gridded sea ice concentration data*, eliminating points within a specified distance of land. This is an implementation of the algorithm described by Eisenman (2010).
* `ice_edge()` is a wrapper function which takes global data and ensures it is in the correct format before passing to `get_ice_edge()`, and returns the (time series of) ice-edge latitudes for each hemisphere. Usually, and for CMIP data in particular, this is the \`user-level' function.
* `split_iel_by_region()` extracts, from the output of `get_ice_edge()` and `ice_edge()`, ice edge points which lie within specified regions (e.g., for getting the ice edge in a certain ocean basin).

### Example usage

Assume we have global, gridded sea ice concentration data `siconc` with time `t`, latitude `lat`, and longitude `lon` coordinates, and grid cell areas, `area`, on a 1&#176; uniform grid and as NumPy arrays such that their shapes: 
```
>>> np.shape(t)
(1980,)  # e.g. CMIP historical simulation monthly means, 185 years * 12 months
>>> np.shape(lat)
(180,)
>>> np.shape(lon)
(360,)
>>> np.shape(area)
(180, 360)
>>> np.shape(siconc)
(1980, 180, 360)
```
Sea ice extent for the whole time series can be calculated (this function also works with un-gridded data):
```
>>> from ice_edge_latitude.diagnostics import sea_ice_extent as sie
>>> from ice_edge_latitude.diagnostics import ice_edge_latitude as iel
>>>
>>> sie_nh, sie_sh = sie.sea_ice_extent(lat, area, siconc)  # northern and southern hemispheres are returned separately
```
Or, e.g., total sea ice area in the marginal ice zone:
```
>>> sia_miz_nh, sia_miz_sh = sie.sea_ice_exent(lat, area, siconc, as_area=True, threshold=[0.15, 0.80])
```
The time series of the latitude of the ice edge (requires gridded data):
```
>>> ie_lat_nh, ie_lat_sh = iel.ice_edge(lat, siconc)
```
The `regions.py` module of the utilities sub-package contains some pre-defined regions for filtering data using `split_iel_by_region`. For example, to get the ice edge points in the south Pacific ocean:
```
>>> from ice_edge_latitude.utilities import regions as reg
>>> ie_lon_sp, ie_lat_sp = iel.split_iel_by_region(lon, ie_lat_sh, region=reg.south_pacific)
```

### Notes
Ensure that the sea ice concentration data and specified threshold parameters are in the same units. For sea ice extent it is necessary to use fractional concentration rather than percentage (which CMIP data often comes in).


## References
Eisenman, I., 2010: Geographic muting of changes in the Arctic sea ice cover. *Geophys. Res. Lett.*, **37**, L16501, [doi:10.1029/2010GL043741](https://doi.org/10.1029/2010GL043741).
