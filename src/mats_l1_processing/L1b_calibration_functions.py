#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 08:46:04 2022

@author: lindamegner
Level 1b calibration functions
"""
import numpy as np

def shift_image(CCDitem, image=None):

    
    """ 
    Shift the images to account for the misalignment. 
    Or rather put the image on a common field of view with all other channels.
    Args:
        CCDitem
        optional image 

    Returns: 
        
        image that has been flipped and shifted
        error_flag

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
    
    #x_minimum=-75
    #y_minimum=0
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
