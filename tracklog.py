#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

"""
file: tracklog.py
created: 1/1/2019
module: tracklog.py
project: pins_tiles
author: Dennis Schmitz

loads and saves tracklogs
"""

import logging
import os
import time

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from .pins import Pin
from .tiles import *

logger = logging.getLogger('tracklog')

__author__ = 'Dennis Schmitz'
__copyright__ = 'Copyright 2020, Dennis Schmitz'
__version__ = '0.0'
__status__ = ''

# Column names from Torque App output and the datatypes
# All float is boring. Also I'm having issues with the pandas read_csv
# so I'm not using it.

# dataset uses "-" string as no-data. if you straight up import without specifying dtype, those columns
# get dtype 'object', where some data is strings. If you use na_values='-' the parser replaces it with
# an ascii '∞' rathen than a numeric NAN. then the float conversion barfs

# so not using them right now. They're all floats anyway and the same effect is saying dtype=float in read_csv

"""
coltypes = {'Longitude': float, 'Latitude': float, 'GPS Speed (Meters/second)': float,
            'Horizontal Dilution of Precision': float,
            'Altitude': float, 'Bearing': float, 'G(x)': float, 'G(y)': float, 'G(z)': float, 'G(calibrated)': float,
            'Absolute Throttle Position B(%)': float, 'Acceleration Sensor(Total)(g)': float,
            'Acceleration Sensor(X axis)(g)': float,
            'Acceleration Sensor(Y axis)(g)': float, 'Acceleration Sensor(Z axis)(g)': float,
            'Accelerator PedalPosition D(%)': float,
            'Accelerator PedalPosition E(%)': float, 'Ambient air temp(°F)': float,
            'Average trip speed(whilst moving only)(mph)': float,
            'Barometer (on Android device)(mb)': float, 'Barometric pressure (from vehicle)(psi)': float,
            'Catalyst Temperature (Bank 1 Sensor 1)(°F)': float, 'Cost per mile/km (Instant)($/m)': float,
            'Cost per mile/km (Trip)($/m)': float,
            'Engine Coolant Temperature(°F)': float, 'Engine kW (At the wheels)(kW)': float, 'Engine Load(%)': float,
            'Engine Load(Absolute)(%)': float, 'Engine RPM(rpm)': float, 'Evap System Vapour Pressure(Pa)': float,
            'Fuel flow rate/minute(gal/min)': float, 'Fuel Level (From Engine ECU)(%)': float,
            'GPS Accuracy(ft)': float,
            'GPS Altitude(ft)': float, 'GPS Bearing(°)': float, 'GPS Latitude(°)': float, 'GPS Longitude(°)': float,
            'GPS Satellites': float,
            'Horsepower (At the wheels)(hp)': float, 'Intake Air Temperature(°F)': float,
            'Intake Manifold Pressure(psi)': float,
            'Miles Per Gallon(Instant)(mpg)': float, 'Miles Per Gallon(Long Term Average)(mpg)': float,
            'Speed (GPS)(mph)': float,
            'Speed (OBD)(mph)': float, 'Torque(Nm)': float, 'Transmission Temperature(Method 1)(°F)': float,
            'Trip average MPG(mpg)': float,
            'Trip Distance(miles)': float, 'Voltage (OBD Adapter)(V)': float, }
"""

# these are used in read_csv to force datetime conversion on these two columns
datecols = ['GPS Time', 'Device Time']


def check_for_strings(tracklist):  # in a list of tracklogs
    """Check for string data in a list of dataframes"""
    if not isinstance(tracklist, list):
        tracklist = [tracklist]
    found = False
    for tl in tracklist:
        df = tl.df
        for col in df:
            if df[col].dtype == "object":
                print("{}:{}".format(df[col].name, df[col].dtype))
                found = True
            if found:
                break
        if found:
            break
    if not found:
        print("No strings in data.")
    else:
        print("Strings found:")
        print(df[col].name)
        print(vars(df[col]))


def load_all_csv_torque_logs(folder, limit=None):
    """Loads all the Torque logs from a given folder and returns a list of the data."""
    folder = os.path.normpath(folder)
    assert os.path.isdir(folder), f"Folder {folder} does not exist"
    logger.debug(folder)
    logs = []
    # get list of csv tracklogs
    for the_file in os.scandir(folder):
        if os.path.splitext(the_file.path)[1].lower() == '.csv':
            logs.append(Tracklog(the_file))
            if limit is not None and len(logs) >= limit:
                logger.warning(f"Reached limit ({limit})")
                break
    return logs


def is_csv(fn):
    if os.path.isfile(fn):
        ext = os.path.splitext(fn)[1]
        if ext.lower() == '.csv':
            return True
    return False


def is_h5(fn):
    if os.path.isfile(fn):
        ext = os.path.splitext(fn)[1]
        if ext.lower() == '.h5':
            return True
    return False


# noinspection PyAttributeOutsideInit
class Tracklog:
    """Loads a single torque log from a directory entry object and returns a Tracklog object with the data"""

    def __init__(self, some_file):
        assert os.path.isfile(some_file)
        self.fn = some_file

    @property
    def metadata(self):
        return {'fdate': self.fdate, 'units': ','.join(self.units)}

    @metadata.setter
    def metadata(self, the_metadata):
        assert isinstance(the_metadata, dict)

    @property
    def units(self):
        if not hasattr(self, '_units'):
            # noinspection PyStatementEffect
            self.df  # forces loading data
            assert hasattr(self, '_units')
        return self._units

    def _from_csv(self):
        # get column names
        with open(self.fn, 'r', encoding='utf-8') as f:
            self.colnames = list(map(str.strip, f.readline().strip().split(',')))

        # get the units and remove from colnames
        # anything in the list and contained in parentheses is a unit for that column
        # after finding, then remove from the column name.

        import re
        # noinspection PyPep8Naming
        TORQUE_UNITS = ['$/m', '%', 'Meters/second', 'Nm', 'Pa', 'V', 'ft', 'g', 'gal/min',
                        'hp', 'kW', 'mb', 'miles', 'mpg', 'mph', 'psi', 'rpm', '°', '°F', '°C']
        units = []
        new_colnames = []
        for name in self.colnames:
            # print(re.findall('\((.*?)\)',s)[-1])
            stuff = re.findall(r'\((.*?)\)', name)  # a list of parenthesized things in the string
            if len(stuff) == 0:
                units.append(None)
                new_colnames.append(name)
            elif stuff[-1] in TORQUE_UNITS:
                units.append(stuff[-1])
                new_colnames.append(name.replace(f'({stuff[-1]})', ''))
            else:
                units.append(None)
                new_colnames.append(name)
        self._units = units
        self.colnames = new_colnames

        # start reading the dataframe skipping row 0 and using our colnames as header. parse the datecols explicitly.
        self._df = pd.read_csv(self.fn, na_values='-', skiprows=[0], header=None, names=self.colnames,
                               parse_dates=datecols, index_col='GPS Time')

        for col in self._df:  # check all the columns for stuff
            if self._df[col].dtype == "object":  # still getting the ascii '∞' instead of NAN so kludge fix this
                self._df[col] = pd.to_numeric(self._df[col], 'coerce')

    def _from_h5(self):
        pass  # stub

    @property
    def df(self):
        if not hasattr(self, '_df'):
            start = time.time()

            if is_csv(self.fn):
                self._from_csv()
            elif is_h5(self.fn):
                self._from_h5()

            self.loadtime = time.time() - start  # not important, using to eval methods to load the dataframe
        return self._df

    @property
    def fdate(self):  # get the date encoded in the filename
        timestamp = None
        path, fn = os.path.split(self.fn)
        fn = os.path.splitext(fn)[0]
        if fn[0:8] == "trackLog":
            the_date, the_time = fn[9:].split("_")
            # print(theDate, theTime)
            the_time = the_time.replace("-", ":")
            the_date_time = "{} {}".format(the_date, the_time)
            timestamp = dateutil.parser.parse(the_date_time)
        return timestamp

    def map_plot(self, *a, **kwargs):
        """
        Plots a track column on a map


        Maps the track's longitude and latitude on a scatter plot with a map background.
        track value shown as color of the plot point
        """

        def rectify(ax, width=15, dpi=100):
            """Change axis aspect for square x and y"""
            # fig.set_dpi(dpi)
            aspect = np.ptp(ax.get_ylim()) / np.ptp(ax.get_xlim())  # y/x
            ax.figure.set_size_inches((width, width * aspect))

        ax = kwargs.get('ax')
        if not kwargs.get('ax'):
            fig, ax = plt.subplots()
            kwargs['ax'] = ax  # pass it down to df.plot()
        else:
            fig = ax.figure

        column = kwargs.get('column')
        if not column:
            column = 'Speed (OBD)'
        else:
            del kwargs['column']  # this is a private param
        if isinstance(column, str):
            column = [column, ]  # turn into list

        detail = kwargs.get('detail')
        if not detail:
            detail = 1
        else:
            del kwargs['detail']

        if not kwargs.get('cmap'):  # The default is bad for speed tracks
            kwargs['cmap'] = 'hot'

        add_colorbar = kwargs.get('add_colorbar')
        if add_colorbar:
            del kwargs['add_colorbar']

        # plot the trace as latitude, longitude scatter with 'column' as color
        kwargs['kind'] = 'scatter'
        kwargs['x'] = 'Longitude'
        kwargs['y'] = 'Latitude'
        kwargs['c'] = column
        self.df.plot(**kwargs)

        add_map_to_ax(ax=ax, detail=detail)  # add a map png as axis background
        rectify(ax)  # fix the screen aspect to unwarp the map

        if not add_colorbar:  # then delete the colorbar created by dataframe.plot
            if len(fig.axes) > 1:
                fig.delaxes(fig.axes[-1])  # last axis is the colorbar

        # plt.axis('off')
        return  # fig, ax

    def plot_on_map(self, param='Speed (OBD)', detail=1, cmap='hot'):
        """@deprecated Plots a trace parameter on a map."""
        lats, lngs = self.df['Latitude'], self.df['Longitude']
        # plot longitude vs latitude using speed as color
        track_bounds = Pin(np.min(lats), np.min(lngs)), Pin(np.max(lats), np.max(lngs))
        track_center = Pin(np.median(lats), np.median(lngs))

        fig = Figure()  # create new figure outside pyplot
        # A canvas must be manually attached to the figure (pyplot would automatically
        # do it).  This is done by instantiating the canvas with the figure as
        # argument.
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # ax.set_aspect('equal')

        # plot the trace as latitude, longitude scatter with 'param' as color
        colors = self.df[param]  # .values
        self.df.plot(kind='scatter', x='Longitude', y='Latitude', ax=ax, c=colors, cmap=cmap)

        # delete the colorbar created by dataframe.plot
        if len(fig.axes) > 1:
            fig.delaxes(fig.axes[-1])  # delete the last one

        # save limits and use them to get a map
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        width = np.ptp(xlim)
        height = np.ptp(ylim)

        self.aspect = height / width

        # fig.set_dpi(100)
        screen_width = 15.
        size = (screen_width, screen_width * self.aspect)
        fig.set_size_inches(size)

        # make lat, lng pins for bounds to get_bounded_map
        # get a map matching the bounds of the plot
        sw = Pin(ylim[0], xlim[0])
        ne = Pin(ylim[1], xlim[1])
        bounds = (sw, ne)
        mapfield = Mapfield(bounds=bounds, detail=detail)
        mapimage = mapfield.image

        ext = *xlim, *ylim  # left, right, bottom, top
        ax.imshow(mapimage, zorder=0, extent=ext, aspect='auto', interpolation='bicubic')

        # plt.axis('off')

        return fig

    @property
    def h5fn(self):
        return os.path.splitext(self.fn)[0] + ".h5"

    def h5store(self, filename=None):
        if not filename:
            filename = self.h5fn
        # df.to_hdf(filentame, key='torque_df', mode='w')
        store = pd.HDFStore(filename)
        store.put('torque_df', self.df)
        store.get_storer('torque_df').attrs.metadata = self.metadata
        store.close()

    def h5load(self, filename=None):
        if not filename:
            filename = self.h5fn
        # self._df = pd.read_hdf(filename)
        store = pd.HDFStore(filename)
        self._df = store['torque_df']
        self.metadata = store.get_storer('torque_df').attrs.metadata
