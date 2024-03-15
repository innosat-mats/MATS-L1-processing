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
    subtract_dark,
    flatfield_calibration,
    get_linearized_image,
    combine_flags,
    make_binary,
    flip_image,
    handle_bad_columns,
    artifact_correction,
    correct_hotpixels,
    correct_single_events
)
from mats_l1_processing.instrument import CCD
from mats_l1_processing.pointing import add_channel_quaternion


# from L1_calibration_functions import get_true_image_old, desmear_true_image_old
#################################################
#       L1 calibration routine                  #
#################################################


def calibrate_all_items(CCDitems, instrument, plot=False):
    import matplotlib.pyplot as plt
    from database_generation.experimental_utils import plot_CCDimage
    

    for CCDitem in CCDitems:
        (
            image_lsb,
            image_bias_sub,
            image_desmeared,
            image_dark_sub,
            image_calib_nonflipped,
            image_calib_flipped,
            image_calibrated,
            errors,
        ) = L1_calibrate(CCDitem, instrument)

        if plot == True:
            fig, ax = plt.subplots(7, 1)
            plot_CCDimage(image_lsb, fig, ax[0], "Original LSB")
            plot_CCDimage(image_bias_sub, fig, ax[1], "Bias subtracted")
            plot_CCDimage(image_desmeared, fig, ax[2], " Desmeared LSB")
            plot_CCDimage(image_dark_sub, fig, ax[3], " Dark current subtracted LSB")
            plot_CCDimage(image_calib_nonflipped, fig, ax[4], " Flat field compensated LSB")
            plot_CCDimage(image_calib_flipped, fig, ax[5], " Flipped images")
            plot_CCDimage(image_calibrated, fig, ax[6], "Artifact corrected LSB (only for nadir)")
            fig.suptitle(CCDitem["channel"])


def L1_calibrate(CCDitem, instrument, force_table: bool = True, return_steps=False):  # This used to take in a calibration_file instread of instrument object

    CCDitem["CCDunit"] =instrument.get_CCD(CCDitem["channel"])
    
    error_bad_column=handle_bad_columns(CCDitem)

    image_lsb = CCDitem["IMAGE"]

    image_se_corrected, error_flags_se  = correct_single_events(CCDitem,image_lsb)
    image_hot_pixel_corrected, error_flags_hp  = correct_hotpixels(CCDitem,image_se_corrected)
    
    # Step 1 and 2: Remove bias and compensate for bad columns, image still in LSB
    image_bias_sub,error_flags_bias = get_true_image(CCDitem,image_hot_pixel_corrected)

    # step 3: correct for non-linearity (image is converted into float??)

    image_linear,error_flags_linearity = get_linearized_image(CCDitem, image_bias_sub, force_table)

    # Step 4: Desmear
    image_desmeared, error_flags_desmear= desmear_true_image(CCDitem, image_linear)
    #    image_desmeared = desmear_true_image(CCDitem)

    # Step 5 Remove dark current
    # TBD: Decide on threshold fro when to use pixel correction (little dark current) and when to use average image correction (large dark current).
    image_dark_sub, error_flags_dark = subtract_dark(CCDitem, image_desmeared)

    # Step 6 The true calibration: All pixels are scaled by the i.e. absolute 
    #and relative calibration factor and their flat_field factor.
    image_flatfielded, error_flags_flatfield= flatfield_calibration(CCDitem, image_dark_sub)
    
    # Flip flipped CCDs
    image_flipped= flip_image(CCDitem, image_flatfielded)
        
    # Step 7 Remove ghost imaging. TBD.
    # error_ghost =  make_binary(np.zeros(CCDitem["IMAGE"].shape,dtype=np.uint16),1)

    image_calibrated = image_flipped

    CCDitem["image_calibrated"] = image_calibrated
    
    # Add channel quaterion to image
    add_channel_quaternion(CCDitem)

    errors = combine_flags([error_bad_column,error_flags_se, error_flags_hp, error_flags_bias,error_flags_linearity,error_flags_desmear,
    error_flags_dark,error_flags_flatfield],
    [1,1,1,1,2,1,3,2]).squeeze()
    
    CCDitem["errors"] = image_calibrated

    if return_steps:
        return image_lsb, image_se_corrected, image_hot_pixel_corrected, image_bias_sub, image_linear, image_desmeared, image_dark_sub, image_flatfielded, image_flipped, image_calibrated, errors
    else:
        return image_calibrated, errors
