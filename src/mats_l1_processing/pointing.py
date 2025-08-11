"""
Author Donal Murtagh
"""

import numpy as np
import xarray as xr
import toml
from scipy.interpolate import bisplev

calibration_data = toml.load(
    "/Users/donal/projekt/SIW/MATS-instrument-data/calibration_data/calibration_data.toml"
)
distortion = xr.open_dataset(calibration_data["pointing"]["splines"])


def add_channel_quaternion(CCDitem):
    """Add channel quaternion to CCDimage This quaternion converts
    from channel coordinated to OHB body coordinates

    Args:
        CCDitem

    """
    CCDitem["qprime"] = CCDitem["CCDunit"].get_channel_quaternion()
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

    h = 6.9  # height of the CCD in mm
    d = 27.6  # width of the CCD in mm

    # selecting effective focal length
    if (ccditem["CCDSEL"]) == 7:  # NADIR channel
        f = 50.6  # effective focal length in mm
    else:  # LIMB channels
        f = 261

    ncskip = ccditem["NCSKIP"]
    try:
        ncbin = ccditem["NCBIN CCDColumns"]
    except:
        ncbin = ccditem["NCBINCCDColumns"]
    nrskip = ccditem["NRSKIP"]
    nrbin = ccditem["NRBIN"]
    ncol = ccditem["NCOL"]  # number of columns in the image MINUS 1

    y_disp = h / (f * 511)
    x_disp = d / (f * 2048)

    if (ccditem["CCDSEL"]) in [1, 3, 5, 6, 7]:
        x_full = 2048 - ncskip - (ncol + 1) * ncbin + ncbin * (xpixel + 0.5)

        # xdeg = np.rad2deg(np.arctan(
        #         x_disp * ((2048 - ncskip - (ncol + 1) * ncbin + ncbin * (xpixel + 0.5))- 2047.0 / 2 )
        # ))
    else:
        x_full = ncskip + ncbin * (xpixel + 0.5)
        # xdeg = np.rad2deg(np.arctan(
        #     x_disp*(ncskip + ncbin * (xpixel+0.5) - 2047. / 2)
        # ))

    y_full = nrskip + nrbin * (ypixel + 0.5)
    # ydeg = np.rad2deg(np.arctan(
    #     y_disp * (nrskip + nrbin * (ypixel + 0.5) - 510. / 2)
    # ))
    channel = ccditem["channel"]
    tckx = [
        distortion[f"{channel}_splinex_t"].values,
        distortion[f"{channel}_splinex_c"].values,
        distortion[f"{channel}_splinex_k"].values,
        3,
        3,
    ]
    tcky = [
        distortion[f"{channel}_spliney_t"].values,
        distortion[f"{channel}_spliney_c"].values,
        distortion[f"{channel}_spliney_k"].values,
        3,
        3,
    ]
    xdistortion = bisplev(y_full, x_full, tckx).squeeze()
    ydistortion = bisplev(y_full, x_full, tcky).squeeze()
    xdeg = np.rad2deg(np.arctan(x_disp * (x_full - 2047.0 / 2 - xdistortion)))
    ydeg = np.rad2deg(np.arctan(y_disp * (y_full - 510.0 / 2 - ydistortion)))

    return xdeg, ydeg
