# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:23:19 2020

@author: bjorn
"""

# %%

# fmt: off
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from mats_l1_processing.L1_calibration_functions import get_true_image
import copy
from mats_l1_processing import read_in_functions
from mats_l1_processing.instrument import nonLinearity
import warnings
# fmt: on


# TO DO: COMBINE THE FOLLOWING TWO FUNCITONS INTO ONE THAT BINS ACCORDING TO
# BOTH FPGA AND ON-CHIP & ROW SETTINGS;


def bin_ref(ref, CCDItem):
    # simple code for binning

    binned,signal_factor = bin_ref_linear(ref,CCDItem)
    return binned, signal_factor

def bin_ref_linear(ref,CCDItem, bias_error = 3):
    #bias error = value subtracted from bias before manual binning is done
    #test image
    nrow, ncol, nrskip, ncskip, nrbin, ncbin, exptime = (
        CCDItem["NROW"],
        CCDItem["NCOL"] + 1,
        CCDItem["NRSKIP"],
        CCDItem["NCSKIP"],
        CCDItem["NRBIN"],
        CCDItem["NCBIN CCDColumns"],
        CCDItem["TEXPMS"],
    )

    #reference image
    nrowr, ncolr, nrskipr, ncskipr, nrbinr, ncbinr, exptimer = (
        ref["NROW"],
        ref["NCOL"] + 1,
        ref["NRSKIP"],
        ref["NCSKIP"],
        ref["NRBIN"],
        ref["NCBIN CCDColumns"],
        ref["TEXPMS"],
    )

    exptimefactor = float(exptime) / float(exptimer)

    if CCDItem["channel"]=='NADIR':
        bias_error = 6

    imgref = ref["IMAGE"]*exptimefactor/ncbinr/nrbinr - bias_error

        # images must cover the same ccd section
    if ncskip == ncskipr and nrskip == nrskipr:
        # declare zero array for row binning
        rowbin = np.zeros([nrow, ncolr])
        for j in range(0, nrow):
            rowbin[j, :] = imgref[j * nrbin : (j + 1) * nrbin, :].sum(axis=0)
            if np.any(rowbin[j, :]<0):    
                warnings.warn('rowbin has negative values')

        binned = np.zeros([nrow, ncol])
        for j in range(0, ncol):
            binned[:, j] = rowbin[:, j * ncbin : (j + 1) * ncbin].sum(axis=1)
            if np.any(binned[:, j] <0):
                warnings.warn('colbin has negative values')

        return binned, exptimefactor*ncbin*nrbin

    else:

        sys.exit("Error: images not from the same CCD region.")

def img_diff(image1, image2):

    return image1 - image2


def get_binning_test_data(
    dirname, channels=[1, 2, 3, 4, 5, 6, 7], test_type_filter="all"
):
    """get data from binning tests. Ignores channels with incomplete tests (not divisible by 4)

    Keyword arguments:
    dirname -- name of directory of data
    channels -- list of channels to consider (default [1,2,3,4,5,6,7])
    test_type -- type of tests to include "all" (default), "col", "row" or "exp"
    """
    CCDitems = []

    # os.chdir(dirname)

    CCDitems = read_in_functions.read_CCDitems(dirname)

    man_tot, inst_tot, channel_tot, test_type_tot = get_binning_test_data_from_CCD_item(
        CCDitems, channels, test_type_filter
    )

    return man_tot, inst_tot, channel_tot, test_type_tot


def get_binning_test_data_from_CCD_item(
    CCDitems,
    channels=[1, 2, 3, 4, 5, 6, 7],
    test_type_filter="all",
    add_bias=False,
    remove_blanks=True,
    n_pixels_to_use=0
):

    drop_first_column = True #Drop first column in analysis since it shows anomoulus behavior for some experiments


    CCDitems_use = []
    IDstrings = []
    binned = []

    # filter on channels
    for i in range(len(CCDitems)):
        if CCDitems[i]["CCDSEL"] in channels:
            CCDitems_use.append(CCDitems[i])

    CCDitems = CCDitems_use
    CCDitems_use = []

    # filter on complete tests
    for channel in channels:
        I = []
        for j in range(len(CCDitems)):
            if CCDitems[j]["CCDSEL"] == channel:
                I.append(j)

        if np.mod(len(I), 4) != 0:
            print("Tests incomplete for channel " + str(channel))
        elif len(I) == 0:
            print("No data for channel " + str(channel))
        else:
            [CCDitems_use.append(CCDitems[i]) for i in I]
    CCDitems = CCDitems_use

    if remove_blanks:
        for i in range(0, len(CCDitems)):
            CCDitems[i]["IMAGE"], _ = get_true_image(CCDitems[i])

    # stack data into 4 arrays one for each measurement type
    CCDl_list = np.copy(CCDitems[0::4])  # long exposure
    CCDs_list = np.copy(CCDitems[1::4])  # short exposure
    CCDr_list = np.copy(CCDitems[2::4])  # reference images
    CCDrs_list = np.copy(CCDitems[3::4])  # reference short (not binned)

    CCDl_sub_img, CCDr_sub_img = [], []
    test_type = np.array([])  # store test type
    for i in range(0, len(CCDs_list)):

        # ASSIGN TEST TYPE, FPGA IS NOT IMPLEMENTED! -olem
        if CCDl_list[i]["TEXPMS"] != CCDr_list[0]["TEXPMS"]:
            test_type = np.append(test_type, "exp")
        elif CCDl_list[i]["NCBIN CCDColumns"] != CCDr_list[0]["NCBIN CCDColumns"]:
            test_type = np.append(test_type, "col")
        elif CCDl_list[i]["NRBIN"] != CCDr_list[0]["NRBIN"]:
            test_type = np.append(test_type, "row")
        else:
            test_type = np.append(test_type, "ref")

        # SUBTRACT DARK IMAGES from long exposure and reference
        CCDl_sub_img.append(
            img_diff(CCDl_list[i]["IMAGE"].copy(), CCDs_list[i]["IMAGE"].copy())
        )
        CCDr_sub_img.append(
            img_diff(CCDr_list[i]["IMAGE"].copy(), CCDrs_list[i]["IMAGE"].copy())
        )

    # DO BINNING SIMULATIONS

    # copy settings from long image (for binning settings)
    bin_input = copy.deepcopy(CCDl_list)

    # replace images with the images with subtrated dark
    for i in range(0, len(CCDl_list)):
        bin_input[i]["IMAGE"] = CCDl_sub_img[i].copy()
        bin_input[i]["TEXPMS"] = bin_input[i]["TEXPMS"]-2000 #2000ms removed since short int image is 2000ms
    # create manually binned images
    signal_factors = []
    for i in range(0, len(CCDr_list)):

        # replace reference image that should be binned manually with one where the dark is removed
        ref = copy.deepcopy(CCDr_list[i])
        ref["IMAGE"] = CCDr_sub_img[i].copy()
        ref["TEXPMS"] = ref["TEXPMS"] - 2000 #2000ms removed since short int image is 2000ms

        # bin reference image according to bin_input settings, where non-linearity from other components are compansated for!
        binned_reference,signal_factor = bin_ref(copy.deepcopy(ref), bin_input[i].copy())

        # adding bias to get the correct values for non-linearity
        if add_bias:
            binned_reference = binned_reference + get_true_image(CCDs_list[i])[0]

        binned.append(binned_reference)
        signal_factors.append(signal_factor)

    man_tot = np.array([])
    inst_tot = np.array([])
    test_type_tot = np.array([])
    channel_tot = np.array([])
    signal_factors_tot = np.array([])

    for i in range(0, len(CCDs_list)):

        if test_type[i] == test_type_filter or (
            (test_type_filter == "all") and (test_type[i] != "ref")
        ):
            # adding bias to get the correct values for non-linearity
            if add_bias:
                inst_bin = CCDl_sub_img[i].copy() + get_true_image(CCDs_list[i])[0]
            else:
                inst_bin = CCDl_sub_img[i].copy()

            if drop_first_column:
                inst_bin = inst_bin[:,1:]
                binned[i] = binned[i][:,1:]

            if n_pixels_to_use>0:
                indexes_to_use = np.random.rand(n_pixels_to_use)*len(inst_bin.flatten())
                
            else:
                indexes_to_use = np.arange(len(inst_bin.flatten()))

            inst_tot = np.append(inst_tot, inst_bin.flatten()[indexes_to_use])
            man_tot = np.append(man_tot, binned[i].flatten()[indexes_to_use])

            test_type_tot = np.append(
                test_type_tot,
                [test_type[i]] * len(inst_bin.flatten()[indexes_to_use]),
            )

            channel_tot = np.append(
                channel_tot, CCDs_list[i]["CCDSEL"] * np.ones(len(inst_bin.flatten()[indexes_to_use]))
            )

            signal_factors_tot = np.append(
                signal_factors_tot,
                [signal_factors[i]] * len(inst_bin.flatten()[indexes_to_use]),
            )

        else:
            pass
            # print("Skipping image " + str(i) + " of type " + test_type[i])

    return man_tot, inst_tot, channel_tot, test_type_tot, signal_factors_tot 


# %%
