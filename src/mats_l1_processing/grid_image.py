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
    x_origo = x_origo + 0.5 
    y_origo = y_origo + 0.5
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
    FOV_X = 6.06
    FOV_Y = 1.52

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

    
    # #Lindas settings as measured in the lab
    #x_pos_vec = {'IR1':75,'IR2':-144,'IR3':-37,'IR4':0,'UV1':-88,'UV2':-156,'NADIR':0} 
    #y_pos_vec = {'IR1': 47,'IR2':76,'IR3':66,'IR4':0,'UV1':13,'UV2':192,'NADIR':0} 
    #Donals settings from star measuremnets
    x_pos_vec = {'IR1':83.47120648,'IR2':-141.29390513,'IR3':-46.07793043,'IR4':0,'UV1':-90.50990343,'UV2':-161.31804504,'NADIR':0} 
    y_pos_vec = {'IR1':48.92274474,'IR2':75.78736229,'IR3':67.06758131,'IR4':0,'UV1':5.37669736,'UV2':188.22050731,'NADIR':0} 
    

    x_pos=round(x_pos_vec[CCDitem['channel']])
    y_pos=round(y_pos_vec[CCDitem['channel']])

    if skip_comp:
        if "flipped" in CCDitem:
            if CCDitem['flipped'] == True:
                x_pos = x_pos + get_CCD_resolution(1,1) - CCDitem['NCSKIP'] - CCDitem['NCBIN FPGAColumns']*CCDitem['NCBIN CCDColumns']*(CCDitem['NCOL']+1) 
            else:
                x_pos = x_pos+CCDitem['NCSKIP']
        elif ((CCDitem['channel']=='IR1') or (CCDitem['channel']=='IR3')
            or (CCDitem['channel']=='UV1') or (CCDitem['channel']=='UV2')):
            x_pos = x_pos + get_CCD_resolution(1,1) - CCDitem['NCSKIP'] - CCDitem['NCBIN FPGAColumns']*CCDitem['NCBIN CCDColumns']*(CCDitem['NCOL']+1) 
        else:
            x_pos = x_pos+CCDitem['NCSKIP']

        y_pos = y_pos+CCDitem['NROWSKIP']

    return x_pos,y_pos


def get_valid_area(CCDitem):

    return  np.zeros(CCDitem['IMAGE'].shape, dtype=np.uint16)


def grid_image(CCDitem,unit):
    """ 
    Adds a grid to the image along the azimuth and elevation angles. Either in units
    of fullframe pixels or degrees. The function also determines 

    Args:
        CCDitem (:obj:'CCDitem'): A CCDitem object
        unit (str): 'pixels' or 'degrees'

    Returns: 
        valid_area (:obj:'np.array'): area of image that is good to use
        x (:obj:'np.array'): shift in horizontal direction (in CCD pixels)
        y (:obj:'np.array'): shift in vertical direction (in CCD pixels)
    """

    x_pos,y_pos = get_shift(CCDitem,skip_comp=True) #get shift for first CCD pixel

    x_origo_full_frame,y_origo_full_frame = get_origo_CCD(unit='pixels')
    
    if unit == 'degrees':
        dtheta,dphi = get_CCD_resolution(CCDitem['NRBIN'],CCDitem['NCBIN FPGAColumns']+CCDitem['NCBIN CCDColumns'])
    elif unit == 'pixels':
        dtheta = CCDitem['NRBIN']
        dphi = CCDitem['NCBIN FPGAColumns']+CCDitem['NCBIN CCDColumns']
    else:
        raise ValueError('Invalid output unit')
    
    x_grid = np.arange(0,CCDitem['NCOL']+1)*dphi+x_pos+dphi/2-x_origo_full_frame
    y_grid = np.arange(0,CCDitem['NROW'])*dtheta+y_pos+dtheta/2-y_origo_full_frame

    CCDitem['azimuth'] = x_grid
    CCDitem['elevation'] = y_grid

    valid_area = get_valid_area(CCDitem)

    return valid_area,x_grid,y_grid

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
    
    
    image_common_fov = np.empty((720,2400))
    error_flag= np.ones(image_common_fov.shape, dtype=np.uint16)
    image_common_fov[:] = np.nan
    image_common_fov[y_rel:y_rel+image.shape[0], x_rel:x_rel+image.shape[1]]=image
    error_flag[y_rel:y_rel+image.shape[0], x_rel:x_rel+image.shape[1]] = 0
    
    return image_common_fov, error_flag