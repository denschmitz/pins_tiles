# pins_tiles
Implements data types for Pin(lat, lng), BaseTile(x, y, zoom), and Tile(x, y, zoom, img)  

A **Pin** is a namedtuple with methods attached to it, and is thus immutable. It allows direct population of the longitude and latitude parameters and also calculates a pin from other sorts of data.  

A **Tile** is a namedtuple with an immutable base class BaseTile. A tile is created with an (x, y, zoom) tuple and adds itself to a queue that fetches the image (currently works with google, openstreetmap and several other free tile services). A tile can also be created from a pin returning the tile containing that pin. If created from a list of pins, it returns a list of tiles that spans the entire area taken by the pins.

Tiles are **cached** to disk with simple algorithm: if the tile is already on disk, use that one; if the tile is not on disk, fetch it and store it to disk. Filename format is x_y_zoom_servicename.png.

Utility function **get_bounded_map()** will build a map from a bound defind by a set of pins and return a map composited from individual tiles and cropped to the bound. Detail level 1 corresponds to the tile zoom where 1 tile can contain all of the pins and each increasing detail level doubles that.

I use **get_bounded_map()** to plot longitude, latitude traces of my car data from a OBD2 adapter on a matplotlib canvas.

![get_bounded_map() example used as a matplotlib plotting canvas](https://raw.githubusercontent.com/denschmitz/pins_tiles/master/example.png)

Utility function **tile_up()** takes a list of tiles and composits them into a mosaic, returning the composite image along with its (lat, long) bounds. This is called by **get_bounded_map()** to create the canvas.

Math follows because there is a (latitude, longitude) to tile coordinate conversion:

From Wolfram: http://mathworld.wolfram.com/MercatorProjection.html

<img src="http://mathworld.wolfram.com/images/eps-gif/MercatorProjection_1000.gif">

The Mercator projection is a map projection that was widely used for navigation since loxodromes are straight lines (although great circles are curved). The following equations place the x-axis of the projection on the equator and the y-axis at longitude <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda_{0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda_{0}" title="\lambda_{0}" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a> is the longitude and <a href="https://www.codecogs.com/eqnedit.php?latex=\phi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /></a> is the latitude.

<a href="https://www.codecogs.com/eqnedit.php?latex=x=\lambda-\lambda_{0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x=\lambda-\lambda_{0}" title="x=\lambda-\lambda_{0}" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=y=ln\left[tan\left(\frac{\pi}{4}&plus;\frac{\phi}{2}\right)\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=ln\left[tan\left(\frac{\pi}{4}&plus;\frac{\phi}{2}\right)\right]" title="y=ln\left[tan\left(\frac{\pi}{4}+\frac{\phi}{2}\right)\right]" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex==\frac{1}{2}ln\left[\frac{(1&plus;sin\&space;\phi)}{(1-sin\&space;\phi)}\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=\frac{1}{2}ln\left[\frac{(1&plus;sin\&space;\phi)}{(1-sin\&space;\phi)}\right]" title="=\frac{1}{2}ln\left[\frac{(1+sin\ \phi)}{(1-sin\ \phi)}\right]" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex==sinh^{-1}\left(tan\&space;\phi\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=sinh^{-1}\left(tan\&space;\phi\right)" title="=sinh^{-1}\left(tan\ \phi\right)" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex==tanh^{-1}\left(sin\&space;\phi\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=tanh^{-1}\left(sin\&space;\phi\right)" title="=tanh^{-1}\left(sin\ \phi\right)" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex==ln\left(tan\&space;\phi&plus;sec\&space;\phi\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=ln\left(tan\&space;\phi&plus;sec\&space;\phi\right)" title="=ln\left(tan\ \phi+sec\ \phi\right)" /></a>

Multiplying by the sphere radius gives a span normalized to <a href="https://www.codecogs.com/eqnedit.php?latex=2\pi&space;R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2\pi&space;R" title="2\pi R" /></a> for a convenient map to the circumference.  

<a href="https://www.codecogs.com/eqnedit.php?latex=-\pi&space;R&space;<&space;x&space;<&space;\pi&space;R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-\pi&space;R&space;<&space;x&space;<&space;\pi&space;R" title="-\pi R < x < \pi R" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=-\frac{\pi}{2}R&space;<&space;y&space;<&space;\frac{\pi}{2}R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-\frac{\pi}{2}R&space;<&space;y&space;<&space;\frac{\pi}{2}R" title="-\frac{\pi}{2}R < y < \frac{\pi}{2}R" /></a>  

The inverse formulas are  
<a href="https://www.codecogs.com/eqnedit.php?latex=\phi=2\&space;tan^{-1}(e^y)-\frac{\pi}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi=2\&space;tan^{-1}(e^y)-\frac{\pi}{2}" title="\phi=2\ tan^{-1}(e^y)-\frac{\pi}{2}" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex==tan^{-1}(sinh\&space;y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=tan^{-1}(sinh\&space;y)" title="=tan^{-1}(sinh\ y)" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex==gd(y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=gd(y)" title="=gd(y)" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=\lambda=x&plus;\lambda_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda=x&plus;\lambda_0" title="\lambda=x+\lambda_0" /></a>  

where gd(y) is the Gudermannian.  

<div id='svgWrapper'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/e/e2/Cylindrical_Projection_basics2.svg'/>
</div>

Web mercator from wikipedia  

<a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;\frac{256}{2\pi}&space;2^{\text{zoom&space;level}}&space;(\lambda&space;&plus;&space;\pi)&space;\text{&space;pixels}\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;\frac{256}{2\pi}&space;2^{\text{zoom&space;level}}&space;(\lambda&space;&plus;&space;\pi)&space;\text{&space;pixels}\\" title="x = \frac{256}{2\pi} 2^{\text{zoom level}} (\lambda + \pi) \text{ pixels}\\" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;\frac{256}{2\pi}&space;2^{\text{zoom&space;level}}&space;\left(\pi&space;-&space;\ln&space;\left[\tan&space;\left(\frac{\pi}{4}&space;&plus;&space;\frac{\varphi}{2}&space;\right)&space;\right]\right)&space;\text{&space;pixels}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;\frac{256}{2\pi}&space;2^{\text{zoom&space;level}}&space;\left(\pi&space;-&space;\ln&space;\left[\tan&space;\left(\frac{\pi}{4}&space;&plus;&space;\frac{\varphi}{2}&space;\right)&space;\right]\right)&space;\text{&space;pixels}" title="y = \frac{256}{2\pi} 2^{\text{zoom level}} \left(\pi - \ln \left[\tan \left(\frac{\pi}{4} + \frac{\varphi}{2} \right) \right]\right) \text{ pixels}" /></a>  

latitude limit which makes square output:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\varphi_{\text{max}}&space;=&space;\left[2\arctan(e^{\pi})&space;-&space;\frac{\pi}{2}\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\varphi_{\text{max}}&space;=&space;\left[2\arctan(e^{\pi})&space;-&space;\frac{\pi}{2}\right]" title="\varphi_{\text{max}} = \left[2\arctan(e^{\pi}) - \frac{\pi}{2}\right]" /></a>  
