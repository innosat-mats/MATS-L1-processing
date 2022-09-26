#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 08:46:04 2022

@author: lindamegner
Level 1b calibration functions
"""
from math import degrees
import numpy as np

def get_full_CCD_pixels():
    #returms number of pixels in a full-frame readout of the CCD
    return 2048,511

def get_center_CCD_pixels():
    #returns the number of pixels from center of CCD to lower left corner
    return 1023,255

def get_origo_CCD(unit='degrees'):
    """ 
    Get position of lower left corner of CCD
    Args:
        rows_binned (int): number of rows binned on the CCD
        total_columns_binned: number of total columns binned (on-chip and fpga)
        

    Returns: 
        dtheta: vertical resolution of pixels in degrees
        dphi: horizontal resolution of pixels in degrees

    """    
    x_origo, y_origo = get_center_CCD_pixels()
    if unit == 'pixels':
        return x_origo,y_origo
    elif unit == 'degrees': 
        dtheta,dphi = get_CCD_resolution(1,1)
        return -x_origo*dphi,-y_origo*dtheta
    else:
        raise ValueError('Invalid output unit')

def get_CCD_resolution(rows_binned = 1,total_columns_binned = 1):
    """ 
    Get angular resolution of MATS CCD
    Args:
        rows_binned (int): number of rows binned on the CCD
        total_columns_binned: number of total columns binned (on-chip and fpga)
        

    Returns: 
        dtheta: vertical resolution of pixels in degrees
        dphi: horizontal resolution of pixels in degrees

    """
    FOV_X = 6
    FOV_Y = 1.5

    x_full,y_full = get_full_CCD_pixels()
    
    DTHETA = FOV_Y/y_full
    DPHI = FOV_X/x_full

    dtheta = DTHETA*rows_binned
    dphi = DPHI*total_columns_binned

    return dtheta,dphi
    


def get_shift(CCDitem,skip_comp=False):
    """ 
    Get the shift to apply to each channel. Reference channel with no shift is 
    channel IR4 using full frame readout and now row or column skipping. Does
    not take into account binning. 

    Args:
        CCDitem (obj): A CCDitem object
        skip_comp (bool): whether to compansate for skipped rows/columns (default: False)

    Returns: 
        x: shift in horizontal direction (in CCD pixels)
        y: shift in vertical direction (in CCD pixels)
    """

    if CCDitem['channel']=='IR1':
        x_pos=-75
        y_pos=47
    elif CCDitem['channel']=='IR2':
        x_pos=144
        y_pos=76      
    elif CCDitem['channel']=='IR3':
        x_pos=37
        y_pos=66 
    elif CCDitem['channel']=='IR4':
        x_pos=0
        y_pos=0
    elif CCDitem['channel']=='UV1':
        #raise Warning('Currently no alignment of UV1')
        x_pos=0
        y_pos=0
    elif CCDitem['channel']=='UV2':
        x_pos=156
        y_pos=192
    elif CCDitem['channel']=='NADIR':
        x_pos=0 # No shifting of NADIR channel
        y_pos=0
    else:
        raise Exception('Unknown channel name', CCDitem['channel'])

    if skip_comp:
        if "flipped" in CCDitem:
            if CCDitem['flipped'] == True:
                x_pos = x_pos + get_CCD_resolution(1,1) - CCDitem['NCSKIP'] - CCDitem['NCBIN FPGAColumns']*CCDitem['NCBIN CCDColumns']*(CCDitem['NCOL']+1) 
            else:
                x_pos = x_pos+CCDitem['NCSKIP']
        elif (CCDitem['channel']=='IR2') or (CCDitem['channel']=='IR4'):
            x_pos = x_pos + get_CCD_resolution(1,1) - CCDitem['NCSKIP'] - CCDitem['NCBIN FPGAColumns']*CCDitem['NCBIN CCDColumns']*(CCDitem['NCOL']+1) 
        else:
            x_pos = x_pos+CCDitem['NCSKIP']

        y_pos = y_pos+CCDitem['NROWSKIP']

    return x_pos,y_pos

def grid_image(CCDitem,unit):
    x_pos,y_pos = get_shift(CCDitem,skip_comp=True) #get shift for first pixel

    x_origo_full_frame,y_origo_full_frame = get_origo_CCD(unit='pixels')
    
    if unit == 'degrees':
        dtheta,dphi = get_CCD_resolution(CCDitem['NRBIN'],CCDitem['NCBIN FPGAColumns']+CCDitem['NCBIN CCDColumns'])
    elif unit == 'pixels':
        dtheta = CCDitem['NRBIN']
        dphi = CCDitem['NCBIN FPGAColumns']+CCDitem['NCBIN CCDColumns']
    else:
        raise ValueError('Invalid output unit')
    #fixme check even and odd numbers for dphi
    
    x_origo = x_pos+x_origo_full_frame+dphi/2
    y_origo = y_pos+y_origo_full_frame+dtheta/2

    x_grid = np.arange(0,CCDitem['NCOL']+1)*dphi-x_origo
    y_grid = np.arange(0,CCDitem['NROW'])*dtheta-y_origo

    return x_grid,y_grid

def shift_image(CCDitem, image=None):
    """ 
    Shift the images to account for the misalignment. 
    Or rather put the image on a common field of view with all other channels.
    Args:
        CCDitem
        optional image 

    Returns: 
        
        image that has been shifted
        error_flag

    """


    x_pos,y_pos = get_shift(CCDitem)

    x_maximum=156
    y_maximum=192
    x_rel=x_maximum-x_pos
    y_rel=y_maximum-y_pos
    
    
    image_common_fov = np.empty((720,2300))
    error_flag= np.ones(image_common_fov.shape, dtype=np.uint16)
    image_common_fov[:] = np.nan
    image_common_fov[y_rel:y_rel+image.shape[0], x_rel:x_rel+image.shape[1]]=image
    error_flag[y_rel:y_rel+image.shape[0], x_rel:x_rel+image.shape[1]] = 0
    
    return image_common_fov, error_flag