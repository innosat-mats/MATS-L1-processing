"""
Author Donal Murtagh
"""



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
    Inputs : CCDitem  , xpixel (can be an array), ypixel (can be an array)
    Outputs xdeg - x offset of the pixel(s) in degrees, ydeg - y offset of the pixels(s) in degrees

    """
    xdisp = 6.06/2047
    ydisp = 1.52/510  # 1.52/510
    ncskip = ccditem['NCSKIP']
    try:
        ncbin = ccditem['NCBIN CCDColumns']
    except:
        ncbin = ccditem['NCBINCCDColumns']
    nrskip = ccditem['NRSKIP']
    nrbin = ccditem['NRBIN']
    ncol = ccditem['NCOL']
    # flipped configuration: clips from right before flip left clip by limiting no. of  columns
    if (ccditem['CCDSEL']) in [1, 3, 5, 6]:
        xdeg = xdisp*((2048-ncskip - (ncol+1)*ncbin +
                      ncbin*(xpixel+0.5)) - 2047./2)
    else:
        xdeg = xdisp*(ncskip + ncbin * (xpixel+0.5) - 2047./2)
    ydeg = ydisp*(nrskip + nrbin * (ypixel+0.5) - 510./2)
    return xdeg, ydeg
