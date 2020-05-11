# -*- coding: utf-8 -*-
"""tests.py
"""

import os
from io import BytesIO
from queue import Queue
from threading import Thread
from collections import namedtuple
from re import match
from functools import reduce
from PIL import Image
from requests import get, Response
from numpy import pi, arctan, exp, log, tan, asarray, arange, extract, ptp, min, max

from .constants import *
from .util import *
from .pins import *
from .tiles import *


def tests():
    """unit tests"""
    # check constants entered correctly
    assert EARTH_RADIUS == 6378137.0, f"Earth Radius: {EARTH_RADIUS}"
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
    """

    # test Pin class
    chicago = Pin(*CHICAGO)  # load from (latitude, longitude)
    chicago_m = Pin.from_merc(*chicago.merc)  # to / from mercator
    chicago_p = Pin.from_pixels(*chicago.pixel_coord(10), 10)  # to / from pix

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

    # for Z in range(13):
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

    # for Z in range(13):
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

    # test Tile class
    chi_tile = Tile.from_pin(chicago, zoom=13)

    # assert Tile.new_tile_q.qsize() == 1
    # assert chi_tile.img is None

    Tile.new_tile_q.join()  # wait for html to arrive

    assert chi_tile.img is not None

    print("yay, winner!")


if __name__ == "__main__":
    tests()
