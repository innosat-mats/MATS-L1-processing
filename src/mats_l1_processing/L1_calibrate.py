#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:41:36 2020

@author: lindamegner
"""

import numpy as np
from mats_l1_processing.L1_calibration_functions import (
    get_true_image,
    desmear_true_image,
    CCD,
    subtract_dark,
    flatfield_calibration,
    get_linearized_image,
    get_linearized_image_parallelized,
    combine_flags,
    make_binary,
    flip_image,
    handle_bad_columns
)

from mats_l1_processing.L1b_calibration_functions import shift_image

# from L1_calibration_functions import get_true_image_old, desmear_true_image_old
#################################################
#       L1 calibration routine                  #
#################################################


def calibrate_all_items(CCDitems, instrument, plot=False):
    import matplotlib.pyplot as plt
    from mats_l1.processing.experimental_utils import plot_CCDimage

    for CCDitem in CCDitems:
        (
            image_lsb,
            image_bias_sub,
            image_desmeared,
            image_dark_sub,
            image_calib_nonflipped,
            image_calibrated,
            image_common_fov,
            errors,
        ) = L1_calibrate(CCDitem, instrument)

        if plot == True:
            fig, ax = plt.subplots(5, 1)
            plot_CCDimage(image_lsb, fig, ax[0], "Original LSB")
            plot_CCDimage(image_bias_sub, fig, ax[1], "Bias subtracted")
            plot_CCDimage(image_desmeared, fig, ax[2], " Desmeared LSB")
            plot_CCDimage(image_dark_sub, fig, ax[3], " Dark current subtracted LSB")
            plot_CCDimage(image_calib_nonflipped, fig, ax[4], " Flat field compensated LSB")
            fig.suptitle(CCDitem["channel"])


def L1_calibrate(CCDitem, instrument): #This used to take in a calibration_file instread of instrument object 

    CCDitem["CCDunit"] =instrument.get_CCD(CCDitem["channel"])

    error_bad_column=handle_bad_columns(CCDitem)

    image_lsb = CCDitem["IMAGE"]
    
    # Step 1 and 2: Remove bias and compensate for bad columns, image still in LSB
    image_bias_sub,error_flags_bias = get_true_image(CCDitem)
    #    image_bias_sub = get_true_image(CCDitem)

    # step 3: correct for non-linearity (image is converted into float??)

    #image_linear,error_flags_linearity = get_linearized_image(CCDitem, image_bias_sub)
    #When uncommenting change the line below too
    
    image_linear = image_bias_sub
    error_flags_linearity =error_flags_bias


    #image_linear

    # Step 4: Desmear
    image_desmeared, error_flags_desmear= desmear_true_image(CCDitem, image_linear)
    #    image_desmeared = desmear_true_image(CCDitem)

    # Step 5 Remove dark current
    # TBD: Decide on threshold fro when to use pixel correction (little dark current) and when to use average image correction (large dark current).
    image_dark_sub, error_flags_dark = subtract_dark(CCDitem, image_desmeared)

    # Step 6 The true calibration: All pixels are scaled by the i.e. absolute 
    #and relative calibration factor and their flat_field factor.
    image_calib_nonflipped, error_flags_flatfield = flatfield_calibration(CCDitem, image_dark_sub)
    
    # Flip image for IR2 and IR4
    image_calibrated= flip_image(CCDitem, image_calib_nonflipped)
    
    #Shift image, i.e. put image on common field of view
    image_common_fov, error_flags_flipnshift = shift_image(CCDitem, image_calibrated)
    
    # Step 7 Remove ghost imaging. TBD.
    error_ghost =  make_binary(np.zeros(CCDitem["IMAGE"].shape,dtype=int),2)

    # Step 8 Transform from LSB to electrons and then to photons. TBD.
    
    CCDitem["image_calibrated"] = image_calib_nonflipped

    error_absolute =  make_binary(np.zeros(CCDitem["IMAGE"].shape,dtype=int),2)

    error_spare = make_binary(np.zeros(CCDitem["IMAGE"].shape,dtype=int),2) #spare error field

    errors = combine_flags([error_bad_column,error_flags_bias,error_flags_linearity,error_flags_desmear,error_flags_dark,error_flags_flatfield,error_ghost,error_absolute,error_spare])
    
    return image_lsb, image_bias_sub, image_desmeared, image_dark_sub, image_calib_nonflipped, image_calibrated, image_common_fov, errors
