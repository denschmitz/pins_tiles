# -*- coding: utf-8 -*-
"""pins.py module

This module implements
    Pin class - immutable (longitude, latitude) on creation
"""

import logging
from collections import namedtuple

import numpy as np

from .constants import *
from .util import *

__all__ = ['Pin']

logger = logging.getLogger('pins')


# Projection transforms This is the core of the spherical to cylindrical conversion to get to tile coords
def project(_lambda, _phi, ang_norm=2.0 * np.pi, R=1.0):
    """Projects spherical (lambda, phi) to cylynder (x, y)"""
    _lambda, _phi = (_lambda / ang_norm) * (2.0 * np.pi), (_phi / ang_norm) * (2.0 * np.pi)
    return (_lambda * R,
            np.log(np.tan(np.pi / 4.0 + _phi / 2.0)) * R)  # x, y


# Inverse Projection, cylindrical to spherical
def project_inv(x, y, ang_norm=2.0 * np.pi, R=1.0):
    """Projects cylynder (x, y) to spherical (lambda, phi)"""
    return ((x / R) / (2.0 * np.pi) * ang_norm,
            (2 * np.arctan(np.exp(y / R)) - np.pi / 2.0) / (2.0 * np.pi) * ang_norm)  # _lambda, _phi


# constructor checks
# (it might be getter to let the functions crash- i haven'e decided)
def check_pixel_coord(pix_coord):
    """Check pixel coord against limits"""
    x, y, zoom = pix_coord
    assert zoom > 0
    assert zoom < MAX_ZOOM
    max_pixel = (2 ** zoom) * TILE_SIZE
    assert 0 <= x <= max_pixel, f'X:{x} outside limits 0 and {max_pixel}'
    assert 0 <= y <= max_pixel, f'Y:{y} outside limits 0 and {max_pixel}'


def check_tile_coord(tile_coord):
    """Check tile coord against limits"""
    x, y, zoom = tile_coord
    assert zoom > 0
    assert zoom < MAX_ZOOM
    max_tile = 2 ** zoom
    assert 0 <= x <= max_tile, f'tile.x:{x} outside limits 0 and {max_tile}'
    assert 0 <= x <= max_tile, f'tile.y:{y} outside limits 0 and {max_tile}'


def check_pin_coord(pin):  # anything that unpacks to (lat, lng)
    """Check (lat, lng) coord against limits"""
    lat, lng = pin
    assert -180.0 <= lng <= 180.0, 'Longitude outside limits -180.0 and 180.0'
    assert -90.0 <= lat <= 90.0, 'Latitude outside limits -90.0 and 90.0'


def check_merc_coord(coord, R=EARTH_RADIUS):  # anything that unpacks to (x, y)
    """checks (x, y) in limits for Spherical Mercator EPSG:900913"""
    x, y = coord
    assert -np.pi * R <= x <= np.pi * R, f'X:{x} outside limits -{np.pi * R} and {np.pi * R}.'
    assert -np.pi * R / 2 <= y <= np.pi * R / 2, f'Y:{y} outside limits -{np.pi * R / 2} and {np.pi * R / 2}.'


def resolution(zoom):
    """returns the mercator meters px resolution at given zoom"""
    return INITIAL_RESOLUTION / (2 ** zoom)  # m/px at given zoom


class Pin(namedtuple('BasePin', 'latitude longitude')):
    """Immutable Pin class
    # the only actual data stored is lat/lng in the base namedtuple class
    # WGS84 (GPS reference)
    #   https://en.wikipedia.org/wiki/World_Geodetic_System#A_new_World_Geodetic_System:_WGS_84

    # spherical (web) mercator projection formula:
    #   http://mathworld.wolfram.com/MercatorProjection.html
    """

    def __eq__(self, other):
        """returns true if two pins are equal"""
        return self.longitude == other.longitude and self.longitude == other.longitude

    def __hash__(self):
        """hash function allows pin to be dictionary key, in sets, etc."""
        return hash((self.latitude, self.longitude))

    @classmethod
    def from_latitude_longitude(cls, latitude=0.0, longitude=0.0):  # this one degenerate and probably useless
        """Creates a point from lat/lon in WGS84"""
        check_pin_coord((latitude, longitude))
        return cls(latitude, longitude)

    from_lat_lng = from_latitude_longitude  # alias

    @classmethod
    def from_merc(cls, x=0.0, y=0.0):  # merc projection with R=EARTH_RADIUS
        """Creates a point from X Y meters in Spherical Mercator EPSG:900913"""
        check_merc_coord((x, y))
        longitude, latitude = project_inv(x, y, ang_norm=360,
                                          R=EARTH_RADIUS)  # note lambda, phi order of long, lat
        return cls(latitude, longitude)

    from_meters = from_merc  # alias

    @classmethod
    def from_merc_web(cls, x, y, zoom):
        """Creates a pin from pixel coordinates and zoom"""
        check_pixel_coord((x, y, zoom))
        scale = TILE_SIZE / (2 * np.pi) * 2 ** zoom  # mercurator coordinates with R=1
        # note y reversal and origin shift to equator/meridian
        longitude, latitude = project_inv(x / scale - np.pi, np.pi - y / scale, ang_norm=360, R=1)
        return cls(latitude, longitude)

        # x, y = pixel_x/scale, pixel_y/scale
        # longitude, latitude = cls._project_inv(x - pi, pi - y, ang_norm=360, R=1)
        # return cls(latitude, longitude)

    from_pixel = from_merc_web  # alias
    from_pixels = from_merc_web  # alias

    @classmethod
    def from_tile_coord(cls, x, y, zoom):
        """Creates a pin from tile coordinates and zoom
        Supposed agnosticism knowing what a 'tile' is achieved by thinking in terms of (x,y,zoom)"""
        check_tile_coord((x, y, zoom))
        scale = 2 ** zoom / (2 * np.pi)
        # note y reversal and origin shift to equator/meridian
        longitude, latitude = project_inv(x / scale - np.pi, np.pi - y / scale, ang_norm=360, R=1)
        return cls(latitude, longitude)

    @classmethod
    def from_tile(cls, tile):  # anything that unpacks to (x, y, zoom)
        """Creates a pin from a tile"""
        return cls.from_tile_coord(*tile)

    @classmethod
    def pin_bounds(cls, pins):  # assume flat array-like
        """Returns two pins encompassing the bounds of a bunch of pins: (sw_pin, ne_pin)"""
        lat, lng = np.asarray(pins).T  # get latitudes and longitudes in flat arrays
        return cls(lat.min(), lng.min()), cls(lat.max(), lng.max())  # (sw_pin, ne_pin)

    @classmethod
    def find_span_zoom(cls, pins):
        """Returns zoom level that spans list of pins
        for sizing only: does not guarantee pins are not split by a tile because pins aren't
        really supposed to know about tiles.
        """
        sw_pin, ne_pin = cls.pin_bounds(pins)
        logger.debug(f'sw:{sw_pin}, ne:{ne_pin}')

        # subtract small lng from large lng and small lat from large lat
        span = np.asarray(ne_pin.merc) - np.asarray(sw_pin.merc)
        logger.debug(f"track span: {span} m")

        # get the resolutions for all the zooms
        zooms = np.arange(MAX_ZOOM)  # check all the zooms
        tile_res = TILE_SIZE * resolution(zooms)

        # get list of zooms big enough to fit all the pins
        good_zooms = np.extract(tile_res > max(span), zooms)
        zoom = max(good_zooms)
        logger.debug(f"zoom: {zoom}, tile span: {tile_res[zoom]}")
        return zoom

    @property
    def latitude_longitude(self):
        """Gets lat/lon in WGS84 (gps standard)
        -90 < latitude < 90, -180 < longitude < 180"""
        return self.latitude, self.longitude

    lat_lng = latitude_longitude  # alias

    @property
    def lat(self):
        return self.latitude

    @property
    def lng(self):
        return self.longitude

    @property
    def lambda_phi(self):
        """Return longitude and latitude in radians (WGS84 origin).
        -pi < lambda < pi, -pi/2 < phi < pi/2"""
        return self.longitude * D2R, self.latitude * D2R  # lambda, phi

    @property
    def merc(self):  # ("real" mercator uses an ellipsoid projection)
        """Return the mercator XY coordinate"""
        return project(self.longitude, self.latitude, ang_norm=360, R=EARTH_RADIUS)

    merc_meters = merc  # alias

    @property
    def merc_norm(self):
        """Return the mercator XY coordinates normalized to 1"""
        return project(self.longitude, self.latitude, ang_norm=360, R=1 / (2 * np.pi))

    @property
    def merc_sphere(self):
        """Return the spherical mercator XY coordinate normalized to 2pi"""
        return project(self.longitude, self.latitude, ang_norm=360, R=1)

    # https://en.wikipedia.org/wiki/Web_Mercator
    def merc_web(self, zoom=0):
        """returns the web mercator pixel coordinates at given zoom
        transforms coordinates so meridian is in center of map and (0,0) is
        upper left corner
        """
        scale = TILE_SIZE / (2 * np.pi) * 2 ** zoom
        x, y = project(self.longitude, self.latitude, ang_norm=360)
        return scale * (x + np.pi), scale * (np.pi - y)  # pixel_x, pixel_y

    @property
    def world(self):
        """returns world coordinate per google javascript api"""
        return self.merc_web()

    def pixel_coord(self, zoom):
        """returns pixel coordinate at given zoom"""
        x, y = self.merc_web(zoom)
        return int(x), int(y)

    pixels = pixel_coord  # alias

    def subpixel_coord(self, zoom):
        """returns subpixel coordinate (within the tile) at given zoom"""
        x, y = self.merc_web(zoom)
        return int(x % TILE_SIZE), int(y % TILE_SIZE)

    def tile_coord(self, zoom):
        """returns tile coordinate at given zoom
        this is the same as the generator data for the Tile class
        """
        x, y = self.merc_web(zoom)
        return int(x / TILE_SIZE), int(y / TILE_SIZE)
