"""
---------------------------------------------------------
DOCUMENTATION FOR:
Regions module of utilities sub-package (regions.py)
---------------------------------------------------------

This module defines some standard geographic regions
useful for extracting subsets of data.

Regions are specified as follows:

    example_region = {'Sub-region 1 name' : pos_1,
                      'Sub-region 2 name' : pos_2,
                      ...
                     }

where

pos_x = [lon_min, lon_max, lat_min, lat_max]

specifies a rectangular boundary of sub-region x. In this
way, arbitrary geometries can be specified or collections
of related regions define as one. See examples below.

Longitudes are in degrees east (i.e. always positive and
between 0 and 360).
---------------------------------------------------------
"""

# Ocean basins:
north_atlantic = {'North-east Atlantic' : [0, 80, 30, 90],
                  'North-west Atlantic' : [260, 360, 30, 90]}

north_pacific = {'North Pacific' : [80, 260, 30, 90]}

south_atlantic = {'South-east Atlantic' : [0, 20, -90, -30],
                  'South-west Atlantic' : [290, 360, -90, -30]}

south_pacific = {'South-east Pacific' : [150, 290, -90, -30]}

south_indian = {'South-east Indian' : [20, 150, -90, -30]}


# Some large lakes and seas which sometimes need to be masked out,
# because ice can erronously appear in the sea ice concentration
# field:
lakes = {'L. Balkhash' : [70, 80, 40, 50],
         'L. Aike'     : [57, 62, 43, 47],
         'Caspian Sea' : [45, 55, 35, 48]}

baltic_sea = {'Sorth Baltic Sea' : [15, 30, 60, 66],
              'South Baltic Sea' : [10, 30, 53, 60]}

black_sea = {'Black Sea' : [26, 41, 40, 47]}
