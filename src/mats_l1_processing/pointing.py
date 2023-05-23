"""
Author Donal Murtagh
"""
import numpy as np
from skyfield.api import wgs84
from skyfield.api import load
from skyfield.units import Distance
from skyfield.framelib import itrs
from skyfield.positionlib import Geocentric, ICRF
from numpy.linalg import norm
import datetime as DT


def add_channel_quaternion(CCDitem):
    """ Add channel quaternion to CCDimage This quaternion converts
    from channel coordinated to OHB body coordinates

    Args:
        CCDitem

    """
    CCDitem['qprime'] = CCDitem["CCDunit"].get_channel_quaternion()
    return





def pix_deg(ccditem, xpixel, ypixel):
    """
    Function to get the x and y angle from a pixel relative to the center of the CCD
        
    Arguments
    ----------
    ccditem : CCDitem
        measurement
    xpixel : int or array[int]
        x coordinate of the pixel(s) in the image
    ypixel : int or array[int]
        y coordinate of the pixel(s) in the image
        
    Returns
    -------
    xdeg : float or array[float]
        angular deviation along the x axis in degrees (relative to the center of the CCD)
    ydeg : float or array[float]
        angular deviation along the y axis in degrees (relative to the center of the CCD) 
    """
    h = 6.9 # height of the CCD in mm
    d = 27.6 # width of the CCD in mm
    # selecting effective focal length
    if (ccditem['CCDSEL']) == 7: # NADIR channel
        f = 50.6 # effective focal length in mm
    else: # LIMB channels
        f = 261    
    
    ncskip = ccditem['NCSKIP']
    try:
        ncbin = ccditem['NCBIN CCDColumns']
    except:
        ncbin = ccditem['NCBINCCDColumns']
    nrskip = ccditem['NRSKIP']
    nrbin = ccditem['NRBIN']
    ncol = ccditem['NCOL'] # number of columns in the image MINUS 1

    y_disp = (h/(f*511))
    x_disp = (d/(f*2048))
  
    if (ccditem['CCDSEL']) in [1, 3, 5, 6, 7]:
        xdeg = np.rad2deg(np.arctan(x_disp*((2048-ncskip - (ncol+1)*ncbin + ncbin*(xpixel+0.5)) - 2047./2)))
    else:
        xdeg = np.rad2deg(np.arctan(x_disp*(ncskip + ncbin * (xpixel+0.5) - 2047./2)))
        
    ydeg = np.rad2deg(np.arctan(y_disp*(nrskip + nrbin * (ypixel+0.5) - 510./2)))

    return xdeg, ydeg


def satpos(ccditem):
    """Function giving the GPS position in lat lon alt..


    Arguments:
        ccditem or dataframe with the 'afsGnssStateJ2000'

    Returns:
        satlat: latitude of satellite (degrees)
        satlon: longitude of satellite (degrees)
        satheight: Altitude in metres

    """
    ecipos= ccditem['afsGnssStateJ2000'][0: 3]
    d = ccditem['EXPDate']
    ts= load.timescale()
    t = ts.from_datetime(d)
    satpo = Geocentric(position_au=Distance(
        m=ecipos).au, t=t)
    satlat, satlong, satheight = satpo.frame_latlon(itrs)
    return (satlat.degrees, satlong.degrees, satheight.m)


def TPpos(ccditem):
    """
    Function giving the GPS TP in lat lon alt..


    Arguments:
        ccditem or dataframe with the 'afsTangentPointECI'

    Returns:
        TPlat: latitude of satellite (degrees)
        TPlon: longitude of satellite (degrees)
        TPheight: Altitude in metres

    """
    eci= ccditem['afsTangentPointECI']
    d = ccditem['EXPDate']
    ts= load.timescale()
    t = ts.from_datetime(d)
    TPpos = Geocentric(position_au=Distance(
        m=eci).au, t=t)
    TPlat, TPlong, TPheight = TPpos.frame_latlon(itrs)
    return (TPlat.degrees, TPlong.degrees, TPheight.m)


def angles(ccditem):
    """
    Function giving various angles..


    Arguments:
        ccditem or dataframe with the 'EXPDate'

    Returns:
        nadir_sza: solar zenith angle at satelite position (degrees)
        TPsza: solar zenith angle at TP position (degrees)
        TPssa: solar scattering angle at TP position (degrees),
        tpLT: Local time at the TP (string)

    """
    planets= load('de421.bsp')
    earth, sun, moon = planets['earth'], planets['sun'], planets['moon']

    d = ccditem['EXPDate']
    ts= load.timescale()
    t = ts.from_datetime(d)
    satlat, satlon, satheight = satpos(ccditem)
    TPlat, TPlon, TPheight= TPpos(ccditem)
    sat_pos= earth + wgs84.latlon(satlat, satlon, elevation_m=satheight)
    sundir= sat_pos.at(t).observe(sun).apparent()
    obs= sundir.altaz()
    nadir_sza= (90-obs[0].degrees)  # nadir solar zenith angle
    TP_pos= earth + wgs84.latlon(TPlat, TPlon, elevation_m=TPheight)
    tpLT= ((d+DT.timedelta(seconds=TPlon/15*60*60)).strftime('%H:%M:%S'))  # 15*60*60 comes from degrees per hour

    FOV= (TP_pos-sat_pos).at(t).position.m
    FOV= FOV/norm(FOV)
    sundir= TP_pos.at(t).observe(sun).apparent()
    obs= sundir.altaz()
    TPsza = (90-obs[0].degrees)
    TPssa = (np.rad2deg(np.arccos(np.dot(FOV,sundir.position.m/norm(sundir.position.m)))))
    return nadir_sza, TPsza, TPssa, tpLT


def nadir_az(ccditem):
    """
    Function giving the solar azimuth angle for the nadir imager  
   
    Arguments:
        ccditem 
    Returns:
        nadir_az: float
            solar azimuth angle at nadir imager (degrees)       
        
    """
    planets=load('de421.bsp')
    earth,sun,moon= planets['earth'], planets['sun'],planets['moon']
   
     
    d = ccditem['EXPDate']
    ts =load.timescale()
    t = ts.from_datetime(d)
    satlat, satlon, satheight = satpos(ccditem)
    TPlat, TPlon, TPheight = TPpos(ccditem)
    
    sat_pos=earth + wgs84.latlon(satlat, satlon, elevation_m=satheight)
    TP_pos=earth + wgs84.latlon(TPlat, TPlon, elevation_m=TPheight)
    sundir=sat_pos.at(t).observe(sun).apparent()
    limbdir = TP_pos.at(t) - sat_pos.at(t)
    obs_limb = limbdir.altaz()
    obs_sun=sundir.altaz()
    nadir_az = (obs_sun[1].degrees - obs_limb[1].degrees) #nadir solar azimuth angle    
    return nadir_az