# -*- coding: utf-8 -*-
"""PinsTiles utility functions"""

from io import BytesIO
from PIL import Image
from requests import Response

from .constants import *

__all__ = ['flatten', 'resolution', 'contenttype', 'is_image', 'get_pic']


def flatten(nl):
    """Flattens a nested list/tuple/set, returns list"""
    if isinstance(nl, (tuple, set)):
        nl = list(nl)

    # noinspection PySimplifyBooleanCheck
    if nl == []:  # don't change this
        return nl

    if isinstance(nl[0], (list, tuple, set)):
        return flatten(list(nl)[0]) + flatten(list(nl)[1:])
    return nl[:1] + flatten(nl[1:])


def resolution(zoom):
    """returns the mercator meters px resolution at given zoom"""
    return INITIAL_RESOLUTION / (2 ** zoom)  # m/px at given zoom


# http utility functions
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
        return ctype, subtype, parameters
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
        try:  # it's lazy, but let Image figure out what it can handle
            return Image.open(BytesIO(resp.content))
        except:
            return None
    else:
        # if isinstance(resp, Response):
        #    print(resp.content[:100]) # replace this with a log
        return None
