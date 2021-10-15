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
    compensate_flatfield,
    get_linearized_image,
)

# from L1_calibration_functions import get_true_image_old, desmear_true_image_old
#################################################
#       L1 calibration routine                  #
#################################################


def calibrate_all_items(CCDitems, plot=False):
    import matplotlib.pyplot as plt
    from LindasCalibrationFunctions import plot_CCDimage

    for CCDitem in CCDitems:
        (
            image_lsb,
            image_bias_sub,
            image_desmeared,
            image_dark_sub,
            image_flatf_comp,
        ) = L1_calibrate(CCDitem)

        if plot == True:
            fig, ax = plt.subplots(5, 1)
            plot_CCDimage(image_lsb, fig, ax[0], "Original LSB")
            plot_CCDimage(image_bias_sub, fig, ax[1], "Bias subtracted")
            plot_CCDimage(image_desmeared, fig, ax[2], " Desmeared LSB")
            plot_CCDimage(image_dark_sub, fig, ax[3], " Dark current subtracted LSB")
            plot_CCDimage(image_flatf_comp, fig, ax[4], " Flat field compensated LSB")
            fig.suptitle(CCDitem["channel"])


def L1_calibrate(CCDitem, calibrationfile):
    global CCDunits

    try:
        CCDunits
    except:
        CCDunits = {}

    # Check  if the CCDunit has been created. It takes time to create it so it should not be created if not needed
    try:
        CCDitem["CCDunit"]
    except:
        try:
            CCDunits[CCDitem["channel"]]
        except:
            CCDunits[CCDitem["channel"]] = CCD(CCDitem["channel"], calibrationfile)
        CCDitem["CCDunit"] = CCDunits[CCDitem["channel"]]

    #  Hack to have no compensation for bad colums at the time. TODO later.
    CCDitem["NBC"] = 0
    CCDitem["BC"] = np.array([])

    # Step0 Compensate for window mode. This should be done first, because it is done in Mikaels software
    # TODO: Cchek if automatic mode and implement it.

    # =============================================================================
    #     #The below should really be doene for old rac files (extracted prior to May2020)
    #     if CCDitem['read_from']=='imgview':
    #         if (CCDitem['WinMode']) <=4:
    #             winfactor=2**CCDitem['WinMode']
    #         elif (CCDitem['WinMode']) ==7:
    #             winfactor=1       # Check that this is what you want!
    #         else:
    #             raise Exception('Undefined Window')
    #         image_lsb=winfactor*CCDitem['IMAGE']
    #     else:
    #         image=_lsb=CCDitem['IMAGE']
    # =============================================================================

    image_lsb = CCDitem["IMAGE"]

    # Step 1 and 2: Remove bias and compensate for bad columns, image still in LSB
    image_bias_sub = get_true_image(CCDitem, image_lsb)
    #    image_bias_sub = get_true_image(CCDitem)

    # step 3: correct for non-linearity
    image_linear = get_linearized_image(CCDitem, image_bias_sub)

    # Step 4: Desmear
    image_desmeared = desmear_true_image(CCDitem, image_bias_sub)
    #    image_desmeared = desmear_true_image(CCDitem)

    # Step 5 Remove dark current
    # TBD: Decide on threshold fro when to use pixel correction (little dark current) and when to use average image correction (large dark current).
    # TBD: The temperature needs to be decided in a better way then taken from the ADC as below.
    # Either read from rac files of temperature sensors or estimated from the top of the image
    image_dark_sub = subtract_dark(CCDitem, image_desmeared)

    # Step 6 Remove flat field of the particular CCD.

    image_flatf_comp = compensate_flatfield(CCDitem, image_dark_sub)
    CCDitem["image_calibrated"] = image_flatf_comp

    # Step 7 Remove ghost imaging. TBD.

    # Step 8 Transform from LSB to electrons and then to photons. TBD.

    return image_lsb, image_bias_sub, image_desmeared, image_dark_sub, image_flatf_comp
