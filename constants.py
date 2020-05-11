# -*- coding: utf-8 -*-
"""
constants.py
"""
from numpy import arctan, pi, exp

EARTH_RADIUS = 6378137.0  # meters
EARTH_CIRCUMFERENCE = 2.0 * pi * EARTH_RADIUS  # m
TILE_SIZE = 256  # px
ORIGIN_SHIFT = EARTH_CIRCUMFERENCE / 2.0  # half way around
INITIAL_RESOLUTION = 2.0 * pi * EARTH_RADIUS / float(TILE_SIZE)  # m/px
LAT_LIMIT = 2.0 * arctan(exp(pi)) - pi / 2.0
D2R = pi / 180.0  # degrees to radians
R2D = 180.0 / pi  # radians to degrees
CHICAGO = (41.850, -87.650)
MAX_ZOOM = 20

__all__ = ['EARTH_RADIUS', 'EARTH_CIRCUMFERENCE', 'TILE_SIZE', 'ORIGIN_SHIFT',
           'INITIAL_RESOLUTION', 'LAT_LIMIT', 'D2R', 'R2D', 'CHICAGO', 'MAX_ZOOM',
           ]
