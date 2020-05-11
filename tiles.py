# -*- coding: utf-8 -*-
"""tiles.py module

This module implements
    Pin class - immutable (longitude, latitude) on creation
    Tile class - immutable (tile_x, tile_y, zoom) on creation
                 mutable img
    Mapfield class

Todo:
    * cache aging mechanism
    * look up GeoJSON
"""

import logging
import os
from collections import namedtuple
from functools import reduce
from queue import Queue
from re import match
from threading import Thread

import numpy as np
import requests
from PIL import Image

from .constants import *
from .pins import Pin
from .util import *

__all__ = ['Tile', 'Mapfield', 'add_map_to_ax']

logger = logging.getLogger('tiles')

__module__ = 'tiles.py'
__author__ = 'Dennis Schmitz'
__copyright__ = 'Copyright 2020, '
__license__ = 'All Rights Reserved'
__version__ = '0.2'
__maintainer__ = 'Dennis Schmitz'
__email__ = 'den@positron.net'
__status__ = 'experimental'

http_headers = {'user-agent': ' : '.join([__module__, __version__, __author__, __email__])}

TILE_CACHE_FOLDER = r'd:\tilecache'
TILE_FETCH_THREADS = 4  # why 4? just because, man.
SERVICE = 'osm'  # open street maps

# maptile utility functions


# background tile fetcher and cache system

tile_urls = {'stamanwatercolor': 'http://c.tile.stamen.com/watercolor/{2}/{0}/{1}.jpg',
             'stamantoner': 'http://a.tile.stamen.com/toner/{2}/{0}/{1}.png',
             'osm': 'https://a.tile.openstreetmap.org/{2}/{0}/{1}.png',
             'wikimedia': 'https://maps.wikimedia.org/osm-intl/{2}/{0}/{1}.png',
             'hillshading': 'http://c.tiles.wmflabs.org/hillshading/{2}/{0}/{1}.png',
             'google': 'https://mts1.google.com/vt/',
             }


def fetch_tile(tile_queue):
    """intended to be started as a daemon thread"""
    while True:
        a_tile = tile_queue.get()
        logger.debug(f'getting {vars(a_tile)}')
        service = a_tile.service
        if not a_tile.nocache:
            a_tile.img = get_cached_image(a_tile, service)
        if a_tile.img is None:
            tile_url = tile_urls[service].format(*a_tile)
            if service is 'google':
                params = {
                    # h = Roads only
                    # m = Standard Roadmap
                    # p = Terrain
                    # r = Roadmap(alternative version)
                    # s = Satellite only
                    # t = Terrain only
                    # y = Hybrid(as seen in the previous entry)
                    'lyrs': a_tile.maptype,  # each tile has it's own maptype
                    'x': a_tile.x,
                    'y': a_tile.y,
                    'z': a_tile.zoom,
                }
                a_tile.http_resp = requests.get(tile_url, params, headers=http_headers)
                a_tile.img = get_pic(a_tile.http_resp)
            else:
                a_tile.http_resp = requests.get(tile_url, headers=http_headers)  # openstreetmap wants special headers
                a_tile.img = get_pic(a_tile.http_resp)
            assert a_tile.img is not None, a_tile.http_resp.request.url + " failed " + str(a_tile.http_resp.status_code)
            put_cached_image(a_tile, service)
        tile_queue.task_done()


def get_cached_image(tile, service):
    """checks if tile from service is in cache and delivers it"""
    fn = "{}_{}_{}_{}.png".format(*tile, service)
    file = os.path.join(TILE_CACHE_FOLDER, fn)
    if os.path.isfile(file):
        return Image.open(file)
    else:
        return None


def put_cached_image(tile, service):
    """saves a tile to the tile cache"""
    fn = "{}_{}_{}_{}.png".format(*tile, service)
    file = os.path.join(TILE_CACHE_FOLDER, fn)
    if tile.img:
        tile.img.save(file)


class PixCoord(namedtuple("XY", "x y")):  # invented to keep X, Y and Row, Col straight
    @classmethod
    def from_rowcol(cls, row=0, col=0):
        """Creates an XY point from rowcol"""
        return cls(col, row)

    @property
    def row(self):
        """returns the y"""
        return self.y

    @property
    def col(self):
        """returns the x"""
        return self.x

    @property
    def rowcol(self):
        """returns the y, x"""
        return self.y, self.x


class Tile(namedtuple('BaseTile', 'x y zoom')):
    """Immutable Tile class

    immutable attributes x, y, and zoom corresponding to a TMS tile.

    mutable attribute img is loaded by a background thread so the main
    program isn't waiting for the internet.

    __hash__ is calculated on (x, y, zoom)
    __eq__ is calculated on (x, y, zoom)
    """

    new_tile_q = Queue()  # tiles waiting for images
    threads = []  # list of threads in case they need killin' (they never did because daemon)
    default_service = SERVICE

    for i in range(TILE_FETCH_THREADS):  # start up some threads to process tiles
        t = Thread(target=fetch_tile, args=[new_tile_q])
        threads.append(t)
        t.daemon = True  # helps with killing threads at the end
        t.start()

    def __new__(cls, *a, **k):
        """Constructor for new Tile assigns values to the namedtuple base class"""
        kk = k.copy()
        if kk.get('nocache'):  # hide the 'foreign' parameter from the upstream namedtuple class
            del kk['nocache']
        # logger.debug(f'pcount:{pcount}')
        if not len(a) + len(kk):  # do nothing if no parameters
            return None
        elif len(a) == 2:  # try to interpret the args as (pin, zoom)
            if isinstance(a[0], Pin) and isinstance(a[1], int):
                zoom = a[1]
                x, y = a[0].tile_coord(zoom)
                a = [x, y, zoom]
                return super().__new__(cls, *a)
            else:
                return None
        else:
            return super().__new__(cls, *a, **kk)

    # noinspection PyArgumentList
    def __init__(self, *a, **k):
        self.nocache = k.get('nocache')
        self._a = a
        self._k = k
        super().__init__()
        self.img = None
        self._service = self.default_service
        self.maptype = 'y'  # subtype for google maps
        self.new_tile_q.put(self)  # add to queue fetching tiles

    def __eq__(self, tile2):
        """returns true if two tiles are equal"""
        return self.x == tile2.x and self.y == tile2.y and self.zoom == tile2.zoom

    def __hash__(self):
        """returns the hash of the underlying tile data"""
        # return hash((self.x, self.y, self.zoom))
        return hash(tuple(self))

    @property
    def service(self):
        return self._service

    @service.setter
    def service(self, val):
        self._service = val
        self.img = None
        self.new_tile_q.put(self)

    @classmethod
    def from_tms(cls, x, y, zoom):
        """Creates a tile from Tile Map Service (TMS) X Y and zoom"""
        max_tile = (2 ** zoom) - 1
        assert 0 <= x <= max_tile, 'TMS X needs to be a value between 0 and (2^zoom) -1.'
        assert 0 <= y <= max_tile, 'TMS Y needs to be a value between 0 and (2^zoom) -1.'
        return cls(x, y, zoom)

    @classmethod
    def from_pin(cls, pins, zoom):
        """Creates tile or tiles to encompase given pin or pins"""
        if isinstance(pins, Pin):  # speed up if passed a single pin
            x, y = pins.tile_coord(zoom)
            return cls(x, y, zoom)
        # else assume a list of pins
        sw_pin, ne_pin = Pin.pin_bounds(pins)  # pin_bounds takes array_like
        sw_tile_coord, ne_tile_coord = sw_pin.tile_coord(zoom), ne_pin.tile_coord(zoom)
        xrange = list(range(sw_tile_coord[0], ne_tile_coord[0] + 1))
        yrange = list(range(ne_tile_coord[1], sw_tile_coord[1] + 1))  # note y tile coords reversed
        tiles = set()
        for x in xrange:
            for y in yrange:
                # print(x,y,zoom)
                tiles.add(cls(x, y, zoom))  # these should all be unique
        return tiles

    from_pins = from_pin  # alias
    from_point = from_pin  # alias
    from_points = from_pin  # alias

    @classmethod
    def from_latitude_longitude(cls, latitude, longitude, zoom):
        """Creates a tile from WGS84 lat/lon and zoom """
        tile_coord = Pin(latitude=latitude, longitude=longitude).tile_coord(zoom)
        return cls(*tile_coord, zoom)

    from_lat_lng = from_latitude_longitude  # alias

    @classmethod
    def from_pixels(cls, pixel_x, pixel_y, zoom):
        """Creates a tile from pixels X Y Z (zoom)"""
        # todo figure out why y axis is being reversed
        x = int(pixel_x / TILE_SIZE)
        y = cls._inv_axis(int(pixel_y / TILE_SIZE), zoom)
        return cls(x, y, zoom)

    @classmethod
    def from_quad_tree(cls, quad_tree):
        """Creates a tile from a Microsoft QuadTree"""
        # todo figure out what a microsoft quadtree is
        assert bool(match('^[0-3]*$', quad_tree)), 'QuadTree value can only consists of the digits 0, 1, 2 and 3.'
        zoom = len(str(quad_tree))
        offset = int(2 ** zoom) - 1
        google_x, google_y = [reduce(lambda result, bit: (result << 1) | bit, bits, 0)
                              for bits in zip(*(reversed(divmod(digit, 2))
                                                for digit in (int(c) for c in str(quad_tree))))]
        return cls(x=google_x, y=(offset - google_y), zoom=zoom)

    @classmethod
    def from_google(cls, google_x, google_y, zoom):
        """Creates a tile from Google format X Y and zoom"""
        # todo figure out why y axis is being reversed
        max_tile = (2 ** zoom) - 1
        assert 0 <= google_x <= max_tile, 'Google X needs to be a value between 0 and (2^zoom) -1.'
        assert 0 <= google_y <= max_tile, 'Google Y needs to be a value between 0 and (2^zoom) -1.'
        return cls(google_x, cls._inv_axis(google_y, zoom), zoom)

    @classmethod
    def from_meters(cls, meter_x, meter_y, zoom):
        """Creates a tile from X Y meters in Spherical Mercator EPSG:900913"""
        tile_coord = Pin.from_meters(meter_x, meter_y).tile_coord(zoom)
        return cls(*tile_coord, zoom)

    def neighbors(self, distance=1):
        """returns a list of neighbor tiles within radius tiles"""
        neighbor_tiles = set([])

        x_range = range(self.x - distance, self.x + distance + 1)
        y_range = range(self.y - distance, self.y + distance + 1)

        for x in x_range:
            for y in y_range:
                if not (x, y) == self.tile_coord:
                    neighbor_tiles.add(Tile(x, y, self.zoom))
        return neighbor_tiles

    @property
    def tile_coord(self):
        """Gets the tile in pyramid from Tile Map Service (TMS)"""
        return self.x, self.y

    tms = tile_coord  # alias

    @property
    def y_inv(self):
        """returns tile y coord inverted on axis"""
        return self._inv_axis(self.y, self.zoom)

    @property
    def quad_tree(self):
        """Gets the tile in the Microsoft QuadTree format, converted from TMS"""
        value = ''
        x = self.x
        y = self.y_inv
        for i in range(self.zoom, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (x & mask) != 0:
                digit += 1
            if (y & mask) != 0:
                digit += 2
            value += str(digit)
        return value

    @property
    def google(self):
        """Gets the tile in the Google format, converted from TMS"""
        return self.x, self.y_inv

    @property
    def bounds(self):
        """Gets the bounds of a tile represented as the most west and south point and the most east and north point"""

        px_n = self.y * TILE_SIZE
        px_s = (self.y + 1) * TILE_SIZE
        px_e = (self.x + 1) * TILE_SIZE
        px_w = self.x * TILE_SIZE

        return (Pin.from_pixel(px_w, px_s, self.zoom),  # sw / min
                Pin.from_pixel(px_e, px_n, self.zoom))  # ne / max

    @staticmethod
    def _inv_axis(x, zoom):
        """inverts an axis at a zoom"""
        return (2 ** zoom - 1) - x

    def _get_tile(self):
        """gets the image tile associated with the tile.
        largely replaced by the daemon"""

        tile_url = "https://mts1.google.com/vt/"
        # tile_url = "http://mt1.google.com/vt/"
        params = {
            'lyrs': 'y',
            'x': self.x,
            'y': self.y,
            'z': self.zoom,
            'src': 'app'}
        self.img = get_pic(requests.get(tile_url, params=params))
        return self.img


def add_map_to_ax(ax, detail=0):
    """Adds a background map to an axes of a plot
    Blows up if the ranges in the ax aren't typical latitude, longitude
    """
    # save limits of the plot and use them to get a map
    #
    # lim = y0 y1 lim.T = y0 x0
    #       x0 x1         y1 x1

    lim = np.asarray([ax.get_ylim(), ax.get_xlim()])
    # get pins for the region bound
    sw = Pin(*lim.T[0])
    ne = Pin(*lim.T[1])
    img = Mapfield(bounds=(sw, ne), detail=detail).image
    ext = *lim[1], *lim[0]  # left, right, bottom, top
    ax.imshow(img, zorder=0, extent=ext, aspect='auto', interpolation='sinc')


class Mapfield:
    """Creates a map field of (presumably) condiguous tiles for now
    Planned to be a pannable map which fetches tiles as necessary into the object.
    Builds up a composite image from all the images in the tiles supplied.
    Requires all tiles to be the same zoom level.
    Does not sanity check out image size.
    """

    def __init__(self, tiles=None, bounds=None, detail=0):  # detail level 0 is TILESIZE over the bound points
        assert (tiles or bounds)  # must have one or other
        self._tiles = tiles
        self._bounds = bounds
        self._detail = detail

        self._X = None
        self._Y = None
        self._zoom = None

        self._tile_canvas = None

        self._prep_tiles()

    def _prep_tiles(self):
        """Prepares the tiles passed into the object"""
        # todo: write this. expected output is a flat iterable.
        # todo: explore turning flatten() into generator

        if self._bounds and not self._tiles:
            # build tile list from bounds
            self._zoom = self._detail + Pin.find_span_zoom(self._bounds)
            self._tiles = Tile.from_pins(self._bounds, self._zoom)  # get the tiles covering the span
            Tile.new_tile_q.join()  # wait for tiles to arrive

        if self._tiles and not self._bounds:
            sw_pin = Pin.from_tile_coord(np.min(self._X), np.max(self._Y) + 1, self._zoom)
            ne_pin = Pin.from_tile_coord(np.max(self._X) + 1, np.min(self._Y), self._zoom)
            self._bounds = sw_pin, ne_pin

        assert all(isinstance(t, Tile) for t in self._tiles), f'{self._tiles}'  # all objects must be tiles
        self._X, self._Y, zooms = np.asarray(list(self._tiles)).T  # asarray won't work on sets. ugh.
        assert all(zooms == zooms[0])  # all zooms must be the same
        self._zoom = zooms[0]

    @property
    def bounds(self):
        return self._bounds

    @property
    def tiles(self):
        return self._tiles

    @property
    def detail(self):
        return self._detail

    @property
    def zoom(self):
        return self._zoom

    @property
    def tile_canvas(self):
        """Makes the output tile_canvas
        ToDo: Sanity check output image size
        ToDo: check tiles? wait for queue?
        """
        if not self._tile_canvas:

            # make blank tile_canvas
            self._tile_canvas = Image.new("RGBA", (
                (np.ptp(self._X) + 1) * TILE_SIZE,
                (np.ptp(self._Y) + 1) * TILE_SIZE))  # (x,y) peak to peak = number of tiles * TILE_SIZE
            logger.debug(f"tile_canvas size:{self._tile_canvas.size}")

            # paint tile_canvas from tiles
            for tile in self.tiles:
                px_x = (tile.x - min(self._X)) * TILE_SIZE
                px_y = (tile.y - min(self._Y)) * TILE_SIZE
                self._tile_canvas.paste(tile.img, (px_x, px_y))

        return self._tile_canvas

    @property
    def tile_bounds(self):
        """Returns tile tile_canvas bounds in form of (sw_pin, ne_pin)"""
        # tile_canvas bounds in pins
        # south is higher Y in tile coords
        # plus one tile n and e because the point will be the sw corner of the tile

        sw_pin = Pin.from_tile_coord(min(self._X), max(self._Y) + 1, self._zoom)
        ne_pin = Pin.from_tile_coord(max(self._X) + 1, min(self._Y), self._zoom)
        return sw_pin, ne_pin

    @property
    def image(self):
        """Outputs a tile_canvas cropped to bounds"""
        px_width, px_height = self.tile_canvas.size

        sw_big, ne_big = self.tile_bounds
        sw, ne = self.bounds

        lat_rng = ne_big.lat - sw_big.lat
        lng_rng = ne_big.lng - sw_big.lng

        lat_lower = sw.lat - sw_big.lat
        lat_upper = ne.lat - sw_big.lat

        lng_left = sw.lng - sw_big.lng
        lng_right = ne.lng - sw_big.lng

        lower = px_height - int(lat_lower / lat_rng * px_height)
        upper = px_height - int(lat_upper / lat_rng * px_height)
        left = int(lng_left / lng_rng * px_width)
        right = int(lng_right / lng_rng * px_width)

        crop_box = left, upper, right, lower
        logger.debug(f'crop_box:{crop_box}')

        return self.tile_canvas.crop(box=crop_box)
