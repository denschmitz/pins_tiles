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
import io

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
__copyright__ = 'Copyright 2022, Dennis Schmitz'
__version__ = '0.2'
__status__ = 'dev'

# Column names from Torque App output and the datatypes
# All float is boring. Also I'm having issues with the pandas read_csv
# so I'm not using it.

# dataset uses "-" string as no-data. if you straight up import without specifying dtype, those columns
# get dtype 'object', where some data is strings. If you use na_values='-' the parser replaces it with
# an ascii '∞' rathen than a numeric NAN. then the float conversion barfs

# so not using them right now. They're all floats anyway and the same effect is saying dtype=float in read_csv

"""
coltypes = {
    'Longitude': float,
    'Latitude': float,
    'GPS Speed (Meters/second)': float,
    'Horizontal Dilution of Precision': float,
    'Altitude': float, 
    'Bearing': float, 
    'G(x)': float, 
    'G(y)': float, 
    'G(z)': float, 
    'G(calibrated)': float,
    'Absolute Throttle Position B(%)': float, 
    'Acceleration Sensor(Total)(g)': float,
    'Acceleration Sensor(X axis)(g)': float,
    'Acceleration Sensor(Y axis)(g)': float, 
    'Acceleration Sensor(Z axis)(g)': float,
    'Accelerator PedalPosition D(%)': float,
    'Accelerator PedalPosition E(%)': float, 
    'Ambient air temp(°F)': float,
    'Average trip speed(whilst moving only)(mph)': float,
    'Barometer (on Android device)(mb)': float, 
    'Barometric pressure (from vehicle)(psi)': float,
    'Catalyst Temperature (Bank 1 Sensor 1)(°F)': float, 
    'Cost per mile/km (Instant)($/m)': float,
    'Cost per mile/km (Trip)($/m)': float,
    'Engine Coolant Temperature(°F)': float, 
    'Engine kW (At the wheels)(kW)': float, 
    'Engine Load(%)': float,
    'Engine Load(Absolute)(%)': float, 
    'Engine RPM(rpm)': float, 
    'Evap System Vapour Pressure(Pa)': float,
    'Fuel flow rate/minute(gal/min)': float, 
    'Fuel Level (From Engine ECU)(%)': float,
    'GPS Accuracy(ft)': float,
    'GPS Altitude(ft)': float, 
    'GPS Bearing(°)': float, 
    'GPS Latitude(°)': float, 
    'GPS Longitude(°)': float,
    'GPS Satellites': float,
    'Horsepower (At the wheels)(hp)': float, 
    'Intake Air Temperature(°F)': float,
    'Intake Manifold Pressure(psi)': float,
    'Miles Per Gallon(Instant)(mpg)': float, 
    'Miles Per Gallon(Long Term Average)(mpg)': float,
    'Speed (GPS)(mph)': float,
    'Speed (OBD)(mph)': float, 
    'Torque(Nm)': float, 
    'Transmission Temperature(Method 1)(°F)': float,
    'Trip average MPG(mpg)': float,
    'Trip Distance(miles)': float,
    'Voltage (OBD Adapter)(V)': float,
    }

GPS Time
Device Time
Longitude
Latitude
GPS Speed (Meters/second)
Horizontal Dilution of Precision
Altitude
Bearing
 G(x)
 G(y)
 G(z)
 G(calibrated)
 Absolute Throttle Position B(%)
Acceleration Sensor(Total)(g)
Acceleration Sensor(X axis)(g)
Acceleration Sensor(Y axis)(g)
Acceleration Sensor(Z axis)(g)
Accelerator PedalPosition D(%)
Accelerator PedalPosition E(%)
Accelerator PedalPosition F(%)
Actual engine % torque(%)
Air Fuel Ratio(Commanded)(:1)
Ambient air temp(°F)
Barometer (on Android device)(mb)
Barometric pressure (from vehicle)(kpa)
Catalyst Temperature (Bank 1 Sensor 1)(°F)
Catalyst Temperature (Bank 1 Sensor 2)(°F)
Catalyst Temperature (Bank 2 Sensor 1)(°F)
Catalyst Temperature (Bank 2 Sensor 2)(°F)
Charge air cooler temperature (CACT)(°F)
Commanded Equivalence Ratio(lambda)
Cost per mile/km (Trip)($/m)
Distance to empty (Estimated)(miles)
Distance travelled since codes cleared(miles)
Distance travelled with MIL/CEL lit(miles)
Drivers demand engine % torque(%)
ECU(7E9): Accelerator PedalPosition D(%)
ECU(7E9): Distance travelled since codes cleared(miles)
ECU(7E9): Distance travelled with MIL/CEL lit(miles)
ECU(7E9): Engine Coolant Temperature(°F)
ECU(7E9): Engine Load(%)
ECU(7E9): Engine RPM(rpm)
ECU(7E9): Speed (OBD)(mph)
ECU(7E9): Voltage (Control Module)(V)
EGR Commanded(%)
EGR Error(%)
Engine Coolant Temperature(°F)
Engine kW (At the wheels)(kW)
Engine Load(%)
Engine Load(Absolute)(%)
Engine Oil Temperature(°F)
Engine reference torque(Nm)
Engine RPM(rpm)
Ethanol Fuel %(%)
Evap System Vapour Pressure(Pa)
Exhaust gas temp Bank 1 Sensor 1(°F)
Exhaust gas temp Bank 1 Sensor 2(°F)
Exhaust gas temp Bank 1 Sensor 3(°F)
Exhaust gas temp Bank 1 Sensor 4(°F)
Exhaust gas temp Bank 2 Sensor 1(°F)
Exhaust gas temp Bank 2 Sensor 2(°F)
Exhaust gas temp Bank 2 Sensor 3(°F)
Exhaust gas temp Bank 2 Sensor 4(°F)
Fuel cost (trip)(cost)
Fuel flow rate/hour(gal/hr)
Fuel flow rate/minute(gal/min)
Fuel Level (From Engine ECU)(%)
Fuel pressure(kpa)
Fuel Rail Pressure(kpa)
Fuel Rail Pressure (relative to manifold vacuum)(kpa)
Fuel Rate (direct from ECU)(L/m)
Fuel Remaining (Calculated from vehicle profile)(%)
Fuel Trim Bank 1 Long Term(%)
Fuel Trim Bank 1 Short Term(%)
Fuel Trim Bank 2 Long Term(%)
Fuel Trim Bank 2 Short Term(%)
Fuel trim Sensor1(%)
Fuel trim Sensor2(%)
Fuel used (trip)(gal)
GPS Accuracy(ft)
GPS Altitude(ft)
GPS Bearing(°)
GPS Latitude(°)
GPS Longitude(°)
GPS Satellites
GPS vs OBD Speed difference(mph)
Horsepower (At the wheels)(hp)
Hybrid Battery Charge (%)(%)
Intake Air Temperature(°F)
Intake Manifold Pressure(kpa)
Kilometers Per Litre(Instant)(kpl)
Kilometers Per Litre(Long Term Average)(kpl)
Litres Per 100 Kilometer(Instant)(l/100km)
Litres Per 100 Kilometer(Long Term Average)(l/100km)
Mass Air Flow Rate(g/s)
Miles Per Gallon(Instant)(mpg)
Miles Per Gallon(Long Term Average)(mpg)
NOx Post SCR(ppm)
NOx Pre SCR(ppm)
O2 Sensor1 Voltage(V)
O2 Sensor2 Voltage(V)
Percentage of City driving(%)
Percentage of Highway driving(%)
Percentage of Idle driving(%)
Relative Accelerator Pedal Position(%)
Relative Throttle Position(%)
Run time since engine start(s)
Speed (GPS)(mph)
Speed (OBD)(mph)
Throttle Position(Manifold)(%)
Timing Advance(°)
Torque(Nm)
Transmission Temperature(Method 2)(°F)
Trip average KPL(kpl)
Trip average Litres/100 KM(l/100km)
Trip average MPG(mpg)
Trip Distance(miles)
Trip distance (stored in vehicle profile)(miles)
Trip Time(Since journey start)(s)
Trip time(whilst moving)(s)
Trip time(whilst stationary)(s)
Turbo Boost & Vacuum Gauge(bar)
Voltage (Control Module)(V)
Voltage (OBD Adapter)(V)
Volumetric Efficiency (Calculated)(%)
DPF Pressure(bar)
DPF Temperature(°F)
Exhaust Pressure(bar)
Fuel trim bank 1 sensor 1(%)
Fuel trim bank 1 sensor 2(%)
Fuel trim bank 1 sensor 3(%)
Fuel trim bank 1 sensor 4(%)
Fuel trim bank 2 sensor 1(%)
Fuel trim bank 2 sensor 2(%)
Fuel trim bank 2 sensor 3(%)
Fuel trim bank 2 sensor 4(%)
O2 Sensor1 Equivalence Ratio
O2 Sensor1 Equivalence Ratio(alternate)
O2 Sensor1 wide-range Voltage
O2 Sensor2 Equivalence Ratio
O2 Sensor2 wide-range Voltage
O2 Sensor3 Equivalence Ratio
O2 Sensor3 wide-range Voltage
O2 Sensor4 Equivalence Ratio
O2 Sensor4 wide-range Voltage
O2 Sensor5 Equivalence Ratio
O2 Sensor5 wide-range Voltage
O2 Sensor6 Equivalence Ratio
O2 Sensor6 wide-range Voltage
O2 Sensor7 Equivalence Ratio
O2 Sensor7 wide-range Voltage
O2 Sensor8 Equivalence Ratio
O2 Sensor8 wide-range Voltage
O2 Volts Bank 1 sensor 1(V)
O2 Volts Bank 1 sensor 2(V)
O2 Volts Bank 1 sensor 3(V)
O2 Volts Bank 1 sensor 4(V)
O2 Volts Bank 2 sensor 1(V)
O2 Volts Bank 2 sensor 2(V)
O2 Volts Bank 2 sensor 3(V)
O2 Volts Bank 2 sensor 4(V)
Transmission Temperature(Method 1)(°F)
Turbo Pressure Control(bar)
"""



tz_str = '''-12 Y
-11 X NUT SST
-10 W CKT HAST HST TAHT TKT
-9 V AKST GAMT GIT HADT HNY
-8 U AKDT CIST HAY HNP PST PT
-7 T HAP HNR MST PDT
-6 S CST EAST GALT HAR HNC MDT
-5 R CDT COT EASST ECT EST ET HAC HNE PET
-4 Q AST BOT CLT COST EDT FKT GYT HAE HNA PYT
-3 P ADT ART BRT CLST FKST GFT HAA PMST PYST SRT UYT WGT
-2 O BRST FNT PMDT UYST WGST
-1 N AZOT CVT EGT
0 Z EGST GMT UTC WET WT
1 A CET DFT WAT WEDT WEST
2 B CAT CEDT CEST EET SAST WAST
3 C EAT EEDT EEST IDT MSK
4 D AMT AZT GET GST KUYT MSD MUT RET SAMT SCT
5 E AMST AQTT AZST HMT MAWT MVT PKT TFT TJT TMT UZT YEKT
6 F ALMT BIOT BTT IOT KGT NOVT OMST YEKST
7 G CXT DAVT HOVT ICT KRAT NOVST OMSST THA WIB
8 H ACT AWST BDT BNT CAST HKT IRKT KRAST MYT PHT SGT ULAT WITA WST
9 I AWDT IRKST JST KST PWT TLT WDT WIT YAKT
10 K AEST ChST PGT VLAT YAKST YAPT
11 L AEDT LHDT MAGT NCT PONT SBT VLAST VUT
12 M ANAST ANAT FJT GILT MAGST MHT NZST PETST PETT TVT WFT
13 FJST NZDT
11.5 NFT
10.5 ACDT LHST
9.5 ACST
6.5 CCT MMT
5.75 NPT
5.5 SLT
4.5 AFT IRDT
3.5 IRST
-2.5 HAT NDT
-3.5 HNT NST NT
-4.5 HLV VET
-9.5 MART MIT'''

tzd = {} # time zone dictionary
for tz_descr in map(str.split, tz_str.split('\n')):
    tz_offset = int(float(tz_descr[0]) * 3600)
    for tz_code in tz_descr[1:]:
        tzd[tz_code] = tz_offset



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


def indexes(iterable, obj):
    return (index for index, elem in enumerate(iterable) if elem == obj)


def groom_columns(cols):
    return cols # stub


#def track_date_parser(*a, **k):
#    k["tzinfos"]=tzd
#    return dateutil.parser.parse(*a, **k)

track_date_parser = lambda x: dateutil.parser.parse(x, tzinfos=tzd)


class Tracklog:
    """Loads a single torque log from a directory entry object and returns a Tracklog object with the data"""

    def __init__(self, some_file):
        assert os.path.isfile(some_file)
        self.path = some_file
        self.df # force loading the file instead of lazy 

    @property
    def metadata(self):
        if not hasattr(self, '_metadata'):
            self._metadata = {}
        return self._metadata

    @property # consider scrapping this because it's stupid
    def units(self):
        if not hasattr(self, '_units'):
            self.df  # forces loading data
            assert hasattr(self, '_units')
        return self._units

    def _from_csv(self):
        print(tzd)

        # read entire file into self._lines
        with open(self.path, 'r', encoding='utf-8') as f:
            self._lines = f.read().splitlines(keepends=False)

        # get column names
        self._colnames = list(map(str.strip, self._lines[0].strip().split(',')))
        assert "GPS Time" in self._colnames
        self.colnames = groom_columns(self._colnames)
        logger.info(f"Found {len(self.colnames)} columns")
        logger.info(f"Found {len(self._lines)} lines")

        # get a list of rows with headers on them to start the ignore list
        skiprows = list(indexes(self._lines, self._lines[0]))
        logger.info(f'Found headers on rows {skiprows}.')

        # get the units and remove from colnames
        # anything in the list and contained in parentheses is a unit for that column
        # after finding, then remove from the column name.

        #import re
        # noinspection PyPep8Naming
        #TORQUE_UNITS = ['$/m', '%', 'Meters/second', 'Nm', 'Pa', 'V', 'ft', 'g', 'gal/min',
        #                'hp', 'kW', 'mb', 'miles', 'mpg', 'mph', 'psi', 'rpm', '°', '°F', '°C']
        #units = []
        #new_colnames = []
        #for name in self.colnames:
        #    # print(re.findall('\((.*?)\)',s)[-1])
        #    # a list of parenthesized things in the string
        #    stuff = re.findall(r'\((.*?)\)', name)
        #    if len(stuff) == 0:
        #        units.append(None)
        #        new_colnames.append(name)
        #    elif stuff[-1] in TORQUE_UNITS:
        #        units.append(stuff[-1])
        #        new_colnames.append(name.replace(f'({stuff[-1]})', ''))
        #    else:
        #        units.append(None)
        #        new_colnames.append(name)
        #self._units = units
        #self.colnames = new_colnames

        # start reading the dataframe skipping rows with headers (usually just the first row) 0
        # and using our colnames as header.
        # 
        # parse the datecols explicitly because torque uses obsolete timezones.

        the_csv = io.StringIO('\n'.join(self._lines))

        self._df = pd.read_csv(the_csv,
                               na_values='-',
                               skiprows=skiprows,
                               header=None,
                               names=self.colnames,
                               parse_dates=datecols,
                               infer_datetime_format=True,
                               cache_dates=True,
                               date_parser=track_date_parser,
                               index_col='GPS Time')


        return

        # check all the columns for stuff
        for col in self._df:
            # still getting the ascii '∞' instead of NAN so kludge fix this
            if self._df[col].dtype == "object": # then there is some non-numeric data there
                self._df[col] = pd.to_numeric(self._df[col], 'coerce')
                logger.info(f"Coerced column {col} to numeric")

    def _from_h5(self):
        pass  # stub

    @property
    def df(self):
        if not hasattr(self, '_df'):
            start = time.time()

            if is_csv(self.path):
                self._from_csv()
            elif is_h5(self.path):
                self._from_h5()

            # not important, using to eval methods to load the dataframe
            self.loadtime = time.time() - start
        return self._df

    @property
    def fdate(self):  # get the date encoded in the filename
        timestamp = None
        path, fn = os.path.split(self.path)
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

        # validate axis and figure
        ax = kwargs.get('ax') # if passed an existing axis, plot into it
        if not ax: # otherwise create it
            fig, ax = plt.subplots()
            kwargs['ax'] = ax  # pass it down to df.plot()
        else:
            fig = ax.figure

        # Look for list of columns to plot in the keywords
        column = kwargs.get('column')
        if not column:
            column = 'Speed (OBD)'
        else:
            del kwargs['column']  # this is a private param that doesn't go on to matplotlib
        if isinstance(column, str):
            column = [column, ]  # turn into list

        # Get map detail level
        detail = kwargs.get('detail')
        if not detail:
            detail = 1
        else:
            del kwargs['detail']

        print(column)
        import sys
        sys.exit()

        if not kwargs.get('cmap'):  # The default is bad for speed tracks
            kwargs['cmap'] = 'hot'

        add_colorbar = kwargs.get('add_colorbar')
        if add_colorbar:
            del kwargs['add_colorbar']

        # plot the trace as latitude, longitude scatter with 'column' as color
        kwargs['kind'] = 'scatter'
        kwargs['x'] = 'Longitude'
        kwargs['y'] = 'Latitude'
        kwargs['c'] = self.df[column]
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
        track_bounds = Pin(np.min(lats), np.min(lngs)), Pin(
            np.max(lats), np.max(lngs))
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
        self.df.plot(kind='scatter', x='Longitude',
                     y='Latitude', ax=ax, c=colors, cmap=cmap)

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
        ax.imshow(mapimage, zorder=0, extent=ext,
                  aspect='auto', interpolation='bicubic')

        # plt.axis('off')

        return fig

    @property
    def h5fn(self):
        return os.path.splitext(self.path)[0] + ".h5"

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
