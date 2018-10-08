# -*- coding: utf-8 -*-
"""PinsTiles module

This module implements
    Pin class - immutable (longitude, latitude) on creation
    Tile class - immutable (tile_x, tile_y, zoom) on creation
                 mutable img

    * tile fetch daemon with 4 processes hoping to parallelize the tile fetches.
    * helper functions

Example:

    from PinsTiles import *
    chicago = Pin(41.850, -87.650)
    chicago_m = Pin.from_merc(*chicago.merc) # translate to/from spherical mercator
    chicago_p = Pin.from_pixels(*chicago.pixel_coord(10), 10)

    x, y = chicago.merc # in meters
    tile_x, tile_y = chicago.tile_coord

    chi_tile = Tile.for_pin(chicago)

    print(chi_tile.bounds)

    chi_neibor_tiles = chi_tile.neibors # a list of 8 encircling neighbors


Attributes:
    module_level_variable1 (int):

Todo:
    * cache aging mechanism
    * look up GeoJSON
"""

#pylint: disable=invalid-name
#pylint: disable=missing-docstring
#pylint: disable=too-many-locals
#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=bare-except

__all__ = ['Pin', 'Tile', 'EARTH_RADIUS', 'CHICAGO', 'TILE_SIZE', 'resolution', 'tile_up',
           'pin_bounds', 'get_bounded_map', 'find_span_zoom']

import os
from io import BytesIO
from queue import Queue
from threading import Thread
from collections import namedtuple
from re import match
from functools import reduce
#from time import sleep

from PIL import Image
from requests import get, Response
from numpy import pi, arctan, exp, log, tan, asarray, arange, extract, ptp, min, max

EARTH_RADIUS = 6378137.0 # meters
EARTH_CIRCUMFERENCE = 2.0 * pi * EARTH_RADIUS # m
TILE_SIZE = 256 # px
ORIGIN_SHIFT = EARTH_CIRCUMFERENCE / 2.0 # half way around
INITIAL_RESOLUTION = 2.0 * pi * EARTH_RADIUS / float(TILE_SIZE) # m/px
LAT_LIMIT = 2.0 * arctan(exp(pi)) - pi/2.0
D2R = pi / 180.0 # degrees to radians
R2D = 180.0 / pi # radians to degrees
CHICAGO = (41.850, -87.650)
MAX_ZOOM = 20
TILE_CACHE_FOLDER = r'd:\tilecache'
TILE_FETCH_THREADS = 4 # why 4? just because, man.
SERVICE = 'osm'


# constructor checks
# (it might be getter to let the functions crash- i haven'e decided)
def check_pixel_coord(coord): # anything that unpacks to (x, y, zoom)
    """Check pixel coord against limits"""
    x, y, zoom = coord
    assert zoom > 0
    assert zoom < MAX_ZOOM
    max_pixel = (2 ** zoom) * TILE_SIZE
    assert 0 <= x <= max_pixel, 'X:{} outside limits 0 and {}'.format(x, max_pixel)
    assert 0 <= y <= max_pixel, 'Y:{} outside limits 0 and {}'.format(y, max_pixel)

def check_tile_coord(tile): # anything that unpacks to (x, y, zoom)
    """Check tile coord against limits"""
    x, y, zoom = tile
    assert zoom > 0
    assert zoom < MAX_ZOOM
    max_tile = 2 ** zoom
    assert 0 <= x <= max_tile, 'tile.x:{} outside limits 0 and {}'.format(x, max_tile)
    assert 0 <= x <= max_tile, 'tile.y:{} outside limits 0 and {}'.format(y, max_tile)

def check_pin_coord(pin): # anything that unpacks to (lat, lng)
    """Check (lat, lng) coord against limits"""
    lat, lng = pin
    assert -180.0 <= lng <= 180.0, 'Longitude outside limits -180.0 and 180.0'
    assert -90.0 <= lat <= 90.0, 'Latitude outside limits -90.0 and 90.0'

def check_merc_coord(coord, R=EARTH_RADIUS): # anything that unpacks to (x, y)
    """checks (x, y) in limits for Spherical Mercator EPSG:900913"""
    x, y = coord
    assert -pi*R <= x <= pi*R, 'X:{1} outside limits -{0} and {0}.'.format(pi*R, x)
    assert -pi*R/2 <= y <= pi*R/2, 'Y:{1} outside limits -{0} and {0}.'.format(pi*R/2, y)

# http utility functions

def resolution(zoom): # make this a lambda?
    """returns the mercator meters px resolution at given zoom"""
    return INITIAL_RESOLUTION / (2 ** zoom) # m/px at given zoom

def contenttype(resp):
    """Breaks out the content string of http response into (type, subtype, parameters)"""
    if isinstance(resp, Response):
        ctype = resp.headers['content-type']
        if '; ' in ctype:
            ctype, parameters = ctype.split('; ')
            ctype, subtype = ctype.split('/')
        else:
            ctype, subtype = ctype.split('/')
            parameters = None
        return type, subtype, parameters
    else:
        return None, None, None

def is_image(resp):
    """Returns true if html request response was an image"""
    try:
        return resp.headers['content-type'].split('/')[0] == 'image'
    except:
        return False

def get_pic(resp):
    """If http response contains an image, load into Image object"""
    if is_image(resp):
        try: # it's lazy, but let Image figure out what it can handle
            return Image.open(BytesIO(resp.content))
        except:
            return None
    else:
        #if isinstance(resp, Response):
        #    print(resp.content[:100]) # replace this with a log
        return None

# pin tile utility functions

def pin_bounds(pins): # assume list/tuple/array of pins
    """Takes a pin or list of pins and returns a tuple bounding pins: (sw_pin, ne_pin)"""
    lat, lng = asarray(pins).T
    return (Pin(lat.min(), lng.min()), Pin(lat.max(), lng.max())) # (sw_pin, ne_pin)

def find_span_zoom(pins):
    """returns zoom level that spans list of pins
    for sizing only: does not guarantee pins are not split by a tile
    """
    sw_pin, ne_pin = pin_bounds(pins)
    #print(sw_pin, ne_pin)
    
    # subtract small lng from large lng and small lat from large lat
    span = asarray(ne_pin.merc) - asarray(sw_pin.merc)
    #print("track span: {} m".format(span))

    # get the resolutions for all the zooms
    zooms = arange(MAX_ZOOM) # all the zooms
    tile_res = TILE_SIZE * resolution(zooms)

    # get list of zooms big enough to hold track and choose largest
    good_zooms = extract(tile_res > max(span), zooms)
    zoom = max(good_zooms)

    #print("zoom {} tile span: {}".format(zoom, tile_res[zoom]))

    return zoom

def tile_up(*args): # take tiles or lists of tiles, or sets of tiles
    """Builds up a composit image from all the images in the tiles supplied.
    Requires all tiles to be the same zoom level.
    Does not sanity check out image size.

    ToDo: Sanity check output image size
    """
    tiles = []
    for t in args:
        tiles.extend(t) # works on sets, i checked

    assert all(isinstance(t, Tile) for t in tiles) # all objects must be tiles
    X, Y, zooms = asarray(tiles).T
    zoom = zooms[0]
    assert all(zooms == zoom) # all zooms must be the same

    sw_pin = Pin.from_tile_coord(min(X), max(Y)+1, zoom) # south is higher Y in tile coords
    ne_pin = Pin.from_tile_coord(max(X)+1, min(Y), zoom) # plus one tile n and e because the point will be the sw corner of the tile

    # make blank canvas
    canvas = Image.new("RGBA", ((ptp(X)+1) * TILE_SIZE, (ptp(Y)+1) * TILE_SIZE)) # (x,y) peak to peak = number of tiles * TILE_SIZE

    #print("canvas size:",canvas.size)

    for tile in tiles:
        #print(vars(tile))
        # tile.img.show()
        
        px_x = (tile.x - min(X)) * TILE_SIZE
        px_y = (tile.y - min(Y)) * TILE_SIZE
        #print(px_x, px_y)
        canvas.paste(tile.img, (px_x, px_y))

    return canvas, (sw_pin, ne_pin)

# maptile utility functions

def get_bounded_map(bounds, detail=0):
    """Builds a map image of entire bounded range by fetching individual tiles
    and creating a composite image that cropped to the area specified.
    bounds can be any collection of pins.
    detail = 0 starts at the zoom level where one tile is large enough to hold entire
    span. (this will be smaller than 256x256).
    """
    zoom = detail + find_span_zoom(bounds)
    tiles = Tile.for_pins(bounds, zoom) # get the tiles covering the span
    #print("tiles:",len(tiles))

    # wait for tiles to arrive
    Tile.new_tile_q.join()

    bigmap, bigmapbounds = tile_up(tiles) # make superset map, then crop

    px_width, px_height = bigmap.size

    sw_big, ne_big = bigmapbounds
    sw, ne = bounds

    lat_rng = ne_big.lat - sw_big.lat
    lng_rng = ne_big.lng - sw_big.lng
    
    lat_lower = sw.lat - sw_big.lat
    lat_upper = ne.lat - sw_big.lat

    lng_left = sw.lng - sw_big.lng
    lng_right = ne.lng - sw_big.lng

    lower = px_height - int(lat_lower/lat_rng * px_height)
    upper = px_height - int(lat_upper/lat_rng * px_height)
    left = int(lng_left/lng_rng * px_width)
    right = int(lng_right/lng_rng * px_width)

    crop_box = left, upper, right, lower
    #print(crop_box)

    return bigmap.crop(box=crop_box)

# background tile fetcher and cache system

tile_urls = {'stamanwatercolor': 'http://c.tile.stamen.com/watercolor/{2}/{0}/{1}.jpg',
             'stamantoner': 'http://a.tile.stamen.com/toner/{2}/{0}/{1}.png',
             'osm': 'https://a.tile.openstreetmap.org/{2}/{0}/{1}.png',
             'wikimedia': 'https://maps.wikimedia.org/osm-intl/{2}/{0}/{1}.png', 
             'hillshading': 'http://c.tiles.wmflabs.org/hillshading/{2}/{0}/{1}.png',
             }

def fetch_tile(tile_queue, service):
    """daemon thread dispatcher"""
    if service == 'google':
        fetch_google_tile(tile_queue, service)
    elif service in tile_urls:
        fetch_generic_tile(tile_queue, service)

def fetch_generic_tile(tile_queue, service):
    """intended to be started as a daemon thread"""
    while True:
        a_tile = tile_queue.get()
        a_tile.img = get_cached_tile(a_tile, service)
        if a_tile.img is None:
            a_tile.http_resp = get(tile_urls[service].format(*a_tile))   
            a_tile.img = get_pic(a_tile.http_resp)
            assert not a_tile.img is None, a_tile.http_resp.request.url + " failed " + str(a_tile.http_resp.status_code)
            put_cached_tile(a_tile, service)
        tile_queue.task_done()

def fetch_google_tile(tile_queue, service):
    """intended to be started as a daemon thread"""
    #tile_url = "http://mt1.google.com/vt/"
    tile_url = "https://mts1.google.com/vt/"
    while True:
        a_tile = tile_queue.get()
        a_tile.img = get_cached_tile(a_tile, service)
        if a_tile.img is None:
            params = {
                'lyrs': 'y',
                'x': a_tile.x,
                'y': a_tile.y,
                'z': a_tile.zoom,
                }
            a_tile.http_resp = get(tile_url, params)
            a_tile.img = get_pic(a_tile.http_resp)
            put_cached_tile(a_tile, service) 
        tile_queue.task_done()

def get_cached_tile(tile, service):
    """checks if tile from service is in cache and delivers it"""
    fn = "{}_{}_{}_{}.png".format(*tile, service)
    file = os.path.join(TILE_CACHE_FOLDER, fn)
    if os.path.isfile(file):
        return Image.open(file)
    else:
        return None

def put_cached_tile(tile, service):
    """saves a tile to the tile cache"""
    fn = "{}_{}_{}_{}.png".format(*tile, service)
    file = os.path.join(TILE_CACHE_FOLDER, fn)
    if tile.img == None:
        return
    else:
        tile.img.save(file)


XY = namedtuple("XY", "x y") # origin upper right image processing

class PixCoord(XY): # invented to keep X, Y and Row, Col straight
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


BasePin = namedtuple('BasePin', 'latitude longitude')

class Pin(BasePin):
    """Immutable Pin class
    # the only actual data stored is lat/lng in the base namedtuple class
    # WGS84 (GPS reference)
    #   https://en.wikipedia.org/wiki/World_Geodetic_System#A_new_World_Geodetic_System:_WGS_84

    # spherical (web) mercator projection formula:
    #   http://mathworld.wolfram.com/MercatorProjection.html
    """

    def __eq__(self, pin2):
        """returns true if two pins are equal"""
        return self.longitude == pin2.longitude and self.longitude == pin2.longitude

    def __hash__(self):
        return hash((self.latitude, self.longitude))

    @classmethod
    def from_latitude_longitude(cls, latitude=0.0, longitude=0.0):
        """Creates a point from lat/lon in WGS84"""
        check_pin_coord((latitude, longitude))
        return cls(latitude, longitude)

    from_lat_lng = from_latitude_longitude # alias

    @classmethod
    def from_merc(cls, x=0.0, y=0.0): # merc projection with R=EARTH_RADIUS
        """Creates a point from X Y meters in Spherical Mercator EPSG:900913"""
        check_merc_coord((x,y))
        longitude, latitude = cls._project_inv(x, y, ang_norm=360, R=EARTH_RADIUS) # note lambda, phi order of long, lat
        return cls(latitude, longitude)

    from_meters = from_merc # alias

    @classmethod
    def from_merc_web(cls, x, y, zoom):
        """Creates a pin from pixel coordinates and zoom"""
        check_pixel_coord((x, y, zoom))
        scale = TILE_SIZE/(2*pi) * 2**zoom # mercurator coordinates with R=1
        # note y reversal and origin shift to equator/meridian
        longitude, latitude = cls._project_inv(x/scale - pi, pi - y/scale, ang_norm=360, R=1)
        return cls(latitude, longitude)

        #x, y = pixel_x/scale, pixel_y/scale
        #longitude, latitude = cls._project_inv(x - pi, pi - y, ang_norm=360, R=1)
        #return cls(latitude, longitude)

    from_pixel = from_merc_web # alias
    from_pixels = from_merc_web # alias

    @classmethod
    def from_tile_coord(cls, x, y, zoom):
        """Creates a pin from tile coordinates and zoom"""
        check_tile_coord((x, y, zoom))
        scale = 2**zoom/(2*pi)
        # note y reversal and origin shift to equator/meridian
        longitude, latitude = cls._project_inv(x/scale - pi, pi - y/scale, ang_norm=360, R=1)
        return cls(latitude, longitude)

    @classmethod
    def from_tile(cls, tile): # anything that unpacks to (x, y, zoom)
        """Creates a pin from a tile"""
        return cls.from_tile_coord(*tile)

    @property
    def latitude_longitude(self):
        """Gets lat/lon in WGS84 (gps standard)
        -90 < latitude < 90, -180 < longitude < 180"""
        return self.latitude, self.longitude

    lat_lng = latitude_longitude # alias

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
        return self.longitude * D2R, self.latitude * D2R # lambda, phi

    @staticmethod
    def _project(_lambda, _phi, ang_norm=2*pi, R=1):
        """Projects spherical (lambda, phi) to cylynder (x, y)"""
        _lambda, _phi = (_lambda/ang_norm) * (2*pi), (_phi/ang_norm) * (2*pi)
        return (_lambda * R,
                log(tan(pi/4.0 + _phi/2.0)) * R) # x, y

    @staticmethod
    def _project_inv(x, y, ang_norm=2*pi, R=1):
        """Projects cylynder (x, y) to spherical (lambda, phi)"""
        return ((x/R)/(2*pi) * ang_norm,
                (2 * arctan(exp(y/R)) - pi/2)/(2*pi) * ang_norm) # _lambda, _phi

    @property
    def merc(self): # ("real" mercator uses an ellipsoid projection)
        """Return the mercator XY coordinate"""
        return self._project(self.longitude, self.latitude, ang_norm=360, R=EARTH_RADIUS)

    merc_meters = merc # alias

    @property
    def merc_norm(self):
        """Return the mercator XY coordinates normalized to 1"""
        return self._project(self.longitude, self.latitude, ang_norm=360, R=1/(2*pi))

    @property
    def merc_sphere(self):
        """Return the spherical mercator XY coordinate normalized to 2pi"""
        return self._project(self.longitude, self.latitude, ang_norm=360, R=1)

    # https://en.wikipedia.org/wiki/Web_Mercator
    def merc_web(self, zoom=0):
        """returns the web mercator pixel coordinates at given zoom
        transforms coordinates so meridian is in center of map and (0,0) is
        upper left corner
        """
        scale = TILE_SIZE/(2*pi) * 2**zoom
        x, y = self._project(self.longitude, self.latitude, ang_norm=360)
        return scale * (x + pi), scale * (pi - y) # pixel_x, pixel_y

    @property
    def world(self):
        """returns world coordinate per google javascript api"""
        return self.merc_web()

    def pixel_coord(self, zoom):
        """returns pixel coordinate at given zoom"""
        x, y = self.merc_web(zoom)
        return int(x), int(y)

    pixels = pixel_coord # alias

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


BaseTile = namedtuple('BaseTile', 'x y zoom') # does not include image or methods

class Tile(BaseTile):
    """Immutable Tile class

    immutable attributes x, y, and zoom corresponding to a TMS tile.

    mutable attribute img is loaded by a background thread so the main
    program isn't waiting for the internet.

    __hash__ is calculated on (x, y, zoom)
    __eq__ is calculated on (x, y, zoom)
    """

    new_tile_q = Queue() # tiles waiting for images
    threads = [] # list of threads in case they need killin'
    for i in range(TILE_FETCH_THREADS): # start up some threads to process tiles
        t = Thread(target=fetch_tile, args=(new_tile_q, SERVICE))
        threads.append(t)
        t.daemon = True
        t.start()

    def __new__(cls, x, y, zoom):
        """New is the place to set the tuple values. we can't directly create
        off the cls variable because there is an img member being added in __init__
        """
        self = super(Tile, cls).__new__(cls, x, y, zoom) # set the namedtuple
        return self

    def __init__(self, x, y, zoom):
        self.img = None # declare the image
        Tile.new_tile_q.put(self) # add to queue fetching tiles

    def __eq__(self, tile2):
        """returns true if two tiles are equal"""
        return self.x == tile2.x and self.y == tile2.y and self.zoom == tile2.zoom

    def __hash__(self):
        """returns the hash of the underlying tile data"""
        return hash((self.x, self.y, self.zoom))

    @classmethod
    def from_tms(cls, x, y, zoom):
        """Creates a tile from Tile Map Service (TMS) X Y and zoom"""
        max_tile = (2 ** zoom) - 1
        assert 0 <= x <= max_tile, 'TMS X needs to be a value between 0 and (2^zoom) -1.'
        assert 0 <= y <= max_tile, 'TMS Y needs to be a value between 0 and (2^zoom) -1.'
        return cls(x, y, zoom)

    @classmethod
    def for_pin(cls, pins, zoom):
        """Creates tile or tiles to encompase given pin or pins"""
        if isinstance(pins, Pin):
            return cls(*pins, zoom)
        # else assume a list of pins
        sw_pin, ne_pin = pin_bounds(pins) # pin_bounds takes array_like
        #print("bounds", sw_pin, ne_pin)
        sw_tile_coord, ne_tile_coord = sw_pin.tile_coord(zoom), ne_pin.tile_coord(zoom)
        xrange = list(range(sw_tile_coord[0], ne_tile_coord[0] + 1))
        yrange = list(range(ne_tile_coord[1], sw_tile_coord[1] + 1)) # note y tile coords reversed
        tiles = set()
        for x in xrange:
            for y in yrange:
                #print(x,y,zoom)
                tiles.add(cls(x, y, zoom)) # these should all be unique
        return tiles

    for_pins = for_pin # alias
    for_point = for_pin # alias
    for_points = for_pin # alias

    @classmethod
    def for_latitude_longitude(cls, latitude, longitude, zoom):
        """Creates a tile from WGS84 lat/lon and zoom """
        tile_coord = Pin(latitude=latitude, longitude=longitude).tile_coord(zoom)
        return cls(*tile_coord, zoom)

    for_lat_lng = for_latitude_longitude # alias

    @classmethod
    def for_pixels(cls, pixel_x, pixel_y, zoom):
        """Creates a tile from pixels X Y Z (zoom)"""
        #todo figure out why y axis is being reversed
        x = int(pixel_x / TILE_SIZE)
        y = cls._inv_axis(int(pixel_y / TILE_SIZE), zoom)
        return cls(x, y, zoom)

    @classmethod
    def from_quad_tree(cls, quad_tree):
        """Creates a tile from a Microsoft QuadTree"""
        #todo figure out what a microsoft quadtree is
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
        #todo figure out why y axis is being reversed
        max_tile = (2 ** zoom) - 1
        assert 0 <= google_x <= max_tile, 'Google X needs to be a value between 0 and (2^zoom) -1.'
        assert 0 <= google_y <= max_tile, 'Google Y needs to be a value between 0 and (2^zoom) -1.'
        return cls(google_x, cls._inv_axis(google_y, zoom), zoom)

    @classmethod
    def for_meters(cls, meter_x, meter_y, zoom):
        """Creates a tile from X Y meters in Spherical Mercator EPSG:900913"""
        tile_coord = Pin.from_meters(meter_x, meter_y).tile_coord(zoom)
        return cls(*tile_coord, zoom)

    def neighbors(self, distance=1):
        """returns a list of neighbor tiles within radius tiles"""
        neighbor_tiles = set([])

        x_range = range(self.x-distance, self.x+distance+1)
        y_range = range(self.y-distance, self.y+distance+1)

        for x in x_range:
            for y in y_range:
                if not (x, y) == self.tile_coord:
                    neighbor_tiles.add(Tile(x, y, self.zoom))
        return neighbor_tiles

    @property
    def tile_coord(self):
        """Gets the tile in pyramid from Tile Map Service (TMS)"""
        return self.x, self.y

    tms = tile_coord# alias

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

        return (Pin.from_pixel(px_w, px_s, self.zoom), # sw / min
                Pin.from_pixel(px_e, px_n, self.zoom)) # ne / max

    @staticmethod
    def _inv_axis(x, zoom):
        """inverts an axis at a zoom"""
        return (2 ** zoom - 1) - x

    def _get_tile(self):
        """gets the image tile associated with the tile.
        largely replaced by the daemon"""

        tile_url = "https://mts1.google.com/vt/"
        #tile_url = "http://mt1.google.com/vt/"
        params = {
            'lyrs': 'y',
            'x': self.x,
            'y': self.y,
            'z': self.zoom,
            'src': 'app'}
        self.img = get_pic(get(tile_url, params=params))
        return self.img


def tests():
    """unit tests"""
    # check constants entered correctly
    assert EARTH_RADIUS == 6378137.0
    assert EARTH_CIRCUMFERENCE == 40075016.68557849
    assert TILE_SIZE == 256
    assert ORIGIN_SHIFT == 20037508.342789244
    assert INITIAL_RESOLUTION == 156543.03392804097
    assert LAT_LIMIT * R2D == 85.0511287798066

    # functions
    print("testing resolution")
    assert resolution(zoom=2) == 39135.75848201024
    assert resolution(zoom=18) == 0.5971642834779395

    """
    print("testing contenttype")
    wiki_png = "https://upload.wikimedia.org/wikipedia/en/thumb/8/80/Wikipedia-logo-v2.svg/526px-Wikipedia-logo-v2.svg.png"
    wiki_svg = "https://upload.wikimedia.org/wikipedia/en/8/80/Wikipedia-logo-v2.svg"
    cnn_url = "http://cnn.com"
    png_resp = get(wiki_png)
    svg_resp = get(wiki_svg)
    cnn_resp = get(cnn_url)
    assert contenttype(png_resp) == ("image", "png", None)
    assert contenttype(cnn_resp) == ("text", "html", "charset=utf-8")

    print("testing is_image")
    assert is_image(png_resp)
    assert is_image(svg_resp)
    assert not is_image(cnn_resp)

    print("testing get_pic")
    assert get_pic(cnn_resp) == None
    assert not get_pic(png_resp) == None
    assert get_pic(svg_resp) == None
    """

    print("testing pin_bounds")
    # ?

    print("testing tile_up")
    # covered by get_bounded_map
    print("testing get_bounded_map")
    sw = Pin(latitude=42.44175005091515, longitude=-83.87692461385079)
    ne = Pin(latitude=42.54204474908484, longitude=-83.51926364614918)
    bounds = (sw, ne)
    img_small = get_bounded_map(bounds, detail=2)
    img_large = get_bounded_map(bounds, detail=4)
    
    assert img_small.size == (521, 198)
    assert img_large.size == (2084, 792)

    img_small.show()
    img_large.show()
    
    return
    
    print("assert resolution(zoom={}) == {}".format(18, resolution(18)))

    """
    # test Pin class
    chicago = Pin(*CHICAGO) # load from (latitude, longitude)
    chicago_m = Pin.from_merc(*chicago.merc) # to / from mercator
    chicago_p = Pin.from_pixels(*chicago.pixel_coord(10), 10) # to / from pix

 

    assert chicago.merc == (-9757153.368030429, 5138536.587247468)
    assert chicago.merc_meters == (-9757153.368030429, 5138536.587247468)
    assert chicago.merc_sphere == (-1.52978108937303, 0.8056485126060271)
    assert chicago.merc_norm == (-0.24347222222222226, 0.12822294317588237)
    assert chicago.world == (65.6711111111111, 95.17492654697412)
    assert chicago.lambda_phi == (-1.52978108937303, 0.730420291959627)
    assert chicago.lat_lng == (41.85, -87.65)
    assert chicago_m.lat_lng == (41.849999999999994, -87.65)
    assert chicago_p.lat_lng == (41.85012764855735, -87.65029907226562)

    assert chicago.pixel_coord(zoom=0) == (65, 95)
    assert chicago.pixel_coord(zoom=1) == (131, 190)
    assert chicago.pixel_coord(zoom=2) == (262, 380)
    assert chicago.pixel_coord(zoom=3) == (525, 761)
    assert chicago.pixel_coord(zoom=4) == (1050, 1522)
    assert chicago.pixel_coord(zoom=5) == (2101, 3045)
    assert chicago.pixel_coord(zoom=6) == (4202, 6091)
    assert chicago.pixel_coord(zoom=7) == (8405, 12182)
    assert chicago.pixel_coord(zoom=8) == (16811, 24364)
    assert chicago.pixel_coord(zoom=9) == (33623, 48729)
    assert chicago.pixel_coord(zoom=10) == (67247, 97459)
    assert chicago.pixel_coord(zoom=11) == (134494, 194918)
    assert chicago.pixel_coord(zoom=12) == (268988, 389836)

    #for Z in range(13):
    #    print("assert chicago.tile_coord(zoom={}) == {}".format(Z, chicago.tile_coord(Z)))

    assert chicago.tile_coord(zoom=0) == (0, 0)
    assert chicago.tile_coord(zoom=1) == (0, 0)
    assert chicago.tile_coord(zoom=2) == (1, 1)
    assert chicago.tile_coord(zoom=3) == (2, 2)
    assert chicago.tile_coord(zoom=4) == (4, 5)
    assert chicago.tile_coord(zoom=5) == (8, 11)
    assert chicago.tile_coord(zoom=6) == (16, 23)
    assert chicago.tile_coord(zoom=7) == (32, 47)
    assert chicago.tile_coord(zoom=8) == (65, 95)
    assert chicago.tile_coord(zoom=9) == (131, 190)
    assert chicago.tile_coord(zoom=10) == (262, 380)
    assert chicago.tile_coord(zoom=11) == (525, 761)
    assert chicago.tile_coord(zoom=12) == (1050, 1522)

    #for Z in range(13):
    #    print("assert chicago.subpixel_coord(zoom={}) == {}".format(Z, chicago.subpixel_coord(Z)))

    assert chicago.subpixel_coord(zoom=0) == (65, 95)
    assert chicago.subpixel_coord(zoom=1) == (131, 190)
    assert chicago.subpixel_coord(zoom=2) == (6, 124)
    assert chicago.subpixel_coord(zoom=3) == (13, 249)
    assert chicago.subpixel_coord(zoom=4) == (26, 242)
    assert chicago.subpixel_coord(zoom=5) == (53, 229)
    assert chicago.subpixel_coord(zoom=6) == (106, 203)
    assert chicago.subpixel_coord(zoom=7) == (213, 150)
    assert chicago.subpixel_coord(zoom=8) == (171, 44)
    assert chicago.subpixel_coord(zoom=9) == (87, 89)
    assert chicago.subpixel_coord(zoom=10) == (175, 179)
    assert chicago.subpixel_coord(zoom=11) == (94, 102)
    assert chicago.subpixel_coord(zoom=12) == (188, 204)
    """

    # test Tile class
    chi_tile = Tile.for_pin(chicago, zoom=13)

    #assert Tile.new_tile_q.qsize() == 1
    #assert chi_tile.img is None

    Tile.new_tile_q.join() # wait for html to arrive

    assert not chi_tile.img is None

    print("yay, winner!")

if __name__ == "__main__":
    tests()
