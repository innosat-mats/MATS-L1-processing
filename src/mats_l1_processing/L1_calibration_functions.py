# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:09:57 2019

@author: franzk, Linda Megner
(Linda has removed some fuctions and added more . Original franzk script is L1_functions.py)


Functions used for MATS L1 processing, based on corresponding MATLAB scripts provided by Georgi Olentsenko and Mykola Ivchenko
The MATLAB script can be found here: https://github.com/OleMartinChristensen/MATS-image-analysis



"""

import numpy as np
import scipy.io
import toml
import scipy.optimize as opt
import pickle
from mats_l1_processing.instrument import nonLinearity as NL
from joblib import Parallel, delayed
import time 
from mats_l1_processing.instrument import CCD


def flip_image(CCDitem, image=None):
    """ Flips the image to account for odd number of mirrors in the light path. 
    Args:
        CCDitem
        optional image 

    Returns: 
        image that has been flipped and shifted

    """
    if image is None:
        image = CCDitem["IMAGE"]
    
    if CCDitem['channel']=='IR2' or CCDitem['channel']=='IR4':
        image=np.fliplr(image)

    return image

def make_binary(flag,bits):
    """Takes in numpy array with (e.g.) dtype=int and returns a numpy 
    array with binary strings.

    Args:
        flag (np.array, dtype=int): numpy array containing the flag
        bits (int): number of error bits represented in the array

    Returns: 
        
        a binary representation of the flag array

    """

    binary_repr_vector = np.vectorize(np.binary_repr)
    
    return binary_repr_vector(flag,bits)


# Utility functions

def combine_flags(flags):
    """Combines binary flags into one binary flag array.
    Args:
        flags (list of np.array of strings ('U<1','U<2','U<4','U<8')):

    Returns: 
        
        total_flag (np.array of stings ('U<16')) the combined flag 

    """
    
    import numpy as np


    
    imsize = flags[0].shape

    total_flag = np.zeros(imsize,dtype='<U16')
    for i in range(total_flag.shape[0]):
        for j in range(total_flag.shape[1]):
            for k in range(len(flags)):
                total_flag[i,j] = total_flag[i,j]+flags[k][i,j]
    
    return total_flag
#%% 
## non-linearity-stuff ##

#%%
def row_sum(true_value_mapped_to_pixels):
    CCD_binned = np.sum(true_value_mapped_to_pixels,axis=0)
    return CCD_binned

def col_sum(true_value_mapped_to_pixels):
    CCD_binned = np.sum(true_value_mapped_to_pixels,axis=0)   
    return CCD_binned

def transfer_function(value_in,non_linearity):
    return non_linearity.get_measured_value(value_in)

def sum_well(true_value_mapped_to_pixels,non_linearity):
    return transfer_function(col_sum(true_value_mapped_to_pixels),non_linearity)

def shift_register(true_value_mapped_to_pixels,non_linearity):
    return transfer_function(row_sum(true_value_mapped_to_pixels),non_linearity)

def single_pixel(true_value_mapped_to_pixels,non_linearity):
    return transfer_function(true_value_mapped_to_pixels,non_linearity)

def total_model(true_value_mapped_to_pixels,p):
    return sum_well(shift_register(single_pixel(true_value_mapped_to_pixels,p[0]),p[1]),p[2])

def total_model_scalar(x,CCD,nrowbin,ncolbin):
    cal_consts = []
    cal_consts.append(CCD.non_linearity_pixel)
    cal_consts.append(CCD.non_linearity_sumrow)
    cal_consts.append(CCD.non_linearity_sumwell)

    true_value_mapped_to_pixels = np.ones((nrowbin,ncolbin))*x/(nrowbin*ncolbin) #expand binned image to pixel values 
    
    return total_model(true_value_mapped_to_pixels,cal_consts) #return modelled value with non-linearity taken into account

def optimize_function(x,CCD,nrowbin,ncolbin,value):
    #x is true value, y is measured value
    y_model = total_model_scalar(x,CCD,nrowbin,ncolbin)
    
    return np.abs(y_model-value)

def test_for_saturation(CCDunit,nrowbin,ncolbin,value):
    
    value_mapped_to_pixels = value/(nrowbin*ncolbin)
    value_mapped_to_shift_register = value/(ncolbin)
    value_mapped_to_summation_well = value

    x = np.nan
    flag = 0 #0 = all ok, 1 = pixel reached non-linearity in pixel, row or column,  3 = pixel reached saturation in pixel, row or column

    if value_mapped_to_pixels>CCDunit.non_linearity_pixel.get_measured_non_lin_important():
        flag = 1
    elif value_mapped_to_shift_register>CCDunit.non_linearity_sumrow.get_measured_non_lin_important():
        flag = 1
    elif value_mapped_to_summation_well>CCDunit.non_linearity_sumwell.get_measured_non_lin_important():
        flag = 1

    if value_mapped_to_pixels>CCDunit.non_linearity_pixel.get_measured_saturation():
            x = CCDunit.non_linearity_pixel.saturation*nrowbin*ncolbin
            flag = 3
    elif value_mapped_to_shift_register>CCDunit.non_linearity_sumrow.get_measured_saturation():
            x = CCDunit.non_linearity_sumrow.saturation*ncolbin
            flag = 3
    elif value_mapped_to_summation_well>CCDunit.non_linearity_sumwell.get_measured_saturation():
            x = CCDunit.non_linearity_sumwell.saturation
            flag = 3

    return flag,x

def check_true_value_max(CCDunit,nrowbin,ncolbin,x_true,flag):
    
    """A method which takes in a CCDunit and binning factors and flags 
    for saturation, and sets value to the saturated value if needed. 

    Args:
        CCDunit (obj): A CCDUnit object
        nrowbin (int): number of rows binned
        ncolbin (int): nuber of cols binned 
        x_true (float): true value of counts
        flag (int): flag of pixel (to be modified)

    Returns: 
        flag (int): flag to indicate high degree of non-linearity and/or saturation
        x (float): true number of counts
    """

    value_mapped_to_pixels = x_true/(nrowbin*ncolbin)
    value_mapped_to_shift_register = x_true/(ncolbin)
    value_mapped_to_summation_well = x_true

    x_true = x_true
    flag = flag #0 = all ok, 1 = pixel reached non-linearity in pixel, row or column,  3 = pixel reached saturation in pixel, row or column

    if value_mapped_to_pixels>CCDunit.non_linearity_pixel.saturation:
            x_true = CCDunit.non_linearity_pixel.saturation*nrowbin*ncolbin
            flag = 3
    elif value_mapped_to_shift_register>CCDunit.non_linearity_sumrow.saturation:
            x_true = CCDunit.non_linearity_sumrow.saturation*ncolbin
            flag = 3
    elif value_mapped_to_summation_well>CCDunit.non_linearity_sumwell.saturation:
            x_true = CCDunit.non_linearity_sumwell.saturation
            flag = 3
    
    return flag,x_true

def inverse_model_real(CCDitem,value,method='BFGS'):
    """A method which takes in a CCDitem and uses the 3 non-linearities in the 
    CCDUnit and the degree of binnning to get the true count (corrected for 
    non-linearity) based on the measured value. This method is slow, so
    if the binning factors are common its faster to pre-calculate a table
    and use *inverse_model_table* instead. 

    Args:
        CCDitem (dict): Dictionary of type CCDItem
        value: Measured value of a pixel
        method: Method to be used in solving the inverse problem

    Returns: 
        x (np.array, dtype=float): true number of counts
        flag (np.array, dtype = int64): flag to indicate high degree of non-linearity and/or saturation
    """

    try:
        CCDunit = CCDitem["CCDunit"]
    except:
        raise Exception("No CCDunit defined for the CCDitem")

    nrowbin = CCDitem["NRBIN"]
    ncolbin = CCDitem["NCBIN CCDColumns"]

    #Check that values are within the linear region:
    
    flag, x = test_for_saturation(CCDunit,nrowbin,ncolbin,value)

    if (flag == 0) or (flag == 1):
        x_hat = opt.minimize(optimize_function,x0=value,args=(CCDunit,nrowbin,ncolbin,value),method=method)
        x = x_hat.x[0]
        flag,x = check_true_value_max(CCDunit,nrowbin,ncolbin,x,flag)


    return x,flag

def inverse_model_table(table,value):
    """Takes in a pre-calculated table and a measured value
    and finds the true number of counts.

    Args:
        table (np.array): lookup table of true counts and flags indexed with counts 
        value (float): measured number of counts

    Returns: 
        x (np.array, dtype=float): true number of counts
        flag (np.array, dtype = int): flag to indicate high degree of non-linearity and/or saturation
    """
    if not (int(table[2,int(value)]) == int(value)):
        raise ValueError('table must be indexed with counts')

    return table[0,int(value)], table[1,int(value)]

def get_linearized_image(CCDitem, image_bias_sub):
    image_linear = np.zeros(image_bias_sub.shape)
    error_flag = np.zeros(image_bias_sub.shape, dtype=np.uint16)

    table = CCDitem['CCDunit'].get_table(CCDitem)
    if table is not None:
        for i in range(image_bias_sub.shape[0]):
            for j in range(image_bias_sub.shape[1]): 
                if image_bias_sub[i,j] > 0:
                    image_linear[i,j],error_flag[i,j] = inverse_model_table(table,image_bias_sub[i,j])
                else:
                    image_linear[i,j] = image_bias_sub[i,j]
    else:
        for i in range(image_bias_sub.shape[0]):
            for j in range(image_bias_sub.shape[1]): 
                if image_bias_sub[i,j] > 0:
                    image_linear[i,j],error_flag[i,j] = inverse_model_real(CCDitem,image_bias_sub[i,j])
                else:
                    image_linear[i,j] = image_bias_sub[i,j]
                
    error_flag = make_binary(error_flag,2)
    
    return image_linear,error_flag

def loop_over_rows(CCDitem,image_bias_sub):
    image_linear = np.zeros(image_bias_sub.shape)
    error_flag = np.zeros(image_bias_sub.shape)
    for j in range(image_bias_sub.shape[0]): 
            image_linear[j],error_flag[j] = inverse_model_real(CCDitem,image_bias_sub[j])

    return image_linear,error_flag

def get_linearized_image_parallelized(CCDitem, image_bias_sub):
    image_linear_list,error_flag = Parallel(n_jobs=4)(delayed( loop_over_rows)(CCDitem,image_bias_sub[i]) for i in range(image_bias_sub.shape[0]))
    return np.array(image_linear_list),np.array(error_flag)

## Flatfield ##

def flatfield_calibration(CCDitem, image=None):
    """Compensates for flatfield.

    Args:
        CCDitem:  dictonary containing CCD image and information
        image (optional) np.array: If this is given then it will be used instead of the image in the CCDitem

    Returns: 
        image_dark_sub (np.array, dtype=float64): true number of counts
        flags (np.array, dtype = uint16): 2 flags to indicate problems with the darc subtractions. 
            Binary array: 1st bit idicates that the dark subtraction renedered a negative value as result, second bit indiates a temperature out of normal range.
    """    
    
    
    if image is None:  
        image = CCDitem["IMAGE"]
    image_flatf_fact = calculate_flatfield(CCDitem)
    # mean = image_flatf_fact[
    #     CCDitem["NRSKIP"] : CCDitem["NRSKIP"] + CCDitem["NROW"],
    #     CCDitem["NCSKIP"] : CCDitem["NCSKIP"] + CCDitem["NCOL"] + 1,
    # ].mean()
    # shape = image_flatf_fact[
    #     CCDitem["NRSKIP"] : CCDitem["NRSKIP"] + CCDitem["NROW"],
    #     CCDitem["NCSKIP"] : CCDitem["NCSKIP"] + CCDitem["NCOL"] + 1,
    # ].shape


    # TODO Absolute relative calibration factors  shoudl be added here
    # Add temperature dependence on flatfield
    image_flatf_comp = (
        image*CCDitem["CCDunit"].cal_fact
        / image_flatf_fact[
            CCDitem["NRSKIP"] : CCDitem["NRSKIP"] + CCDitem["NROW"],
            CCDitem["NCSKIP"] : CCDitem["NCSKIP"] + CCDitem["NCOL"] + 1,
        ]
    )
    # rows,colums Note that nrow always seems to be implemented as +1 already, whereas NCOL does not, hence the missing '+1' in the column calculation /LM201204


    error_flag= np.zeros(image.shape, dtype=np.uint16)
    error_flag[image_flatf_comp<0] = 1 # Flag for negative value
    
    error_flag = make_binary(error_flag,2)

    return image_flatf_comp, error_flag


def calculate_flatfield(CCDitem):
    """Calculates the flatfield factor for binned images, otherwise simply returns 
    the flatfield of the channel. This factor should be diveded with to comensate for flatfield.

    Args:
        CCDitem:  dictonary containing CCD image and information


    Returns: 
        image_flatf_fact: np.array of the same size as the binned image, 
        which should be divided with to compensate for flatfield
    """
    try:
        CCDunit = CCDitem["CCDunit"]
    except:
        raise Exception("No CCDunit defined for the CCDitem")
    image_flatf = CCDunit.flatfield(CCDitem["GAIN Mode"])

    # area_ymin = 350
    # area_ymax = 400
    # area_xmin = 524
    # area_xmax = 1523    
    # meanff = np.mean(image_flatf[area_ymin:area_ymax, area_xmin + 1 : area_xmax + 1])  # Note that this is the mean of the full flatfield , not of the part of the image used.
    if (
        CCDitem["NCBIN CCDColumns"] > 1 or CCDitem["NCBIN FPGAColumns"] > 1
    ):  # Or row binning
        image_flatf = meanbin_image_with_BC(CCDitem, image_flatf)
    image_flatf_fact = image_flatf  # /meanff #Already scaled wheen binned and


    return image_flatf_fact


def subtract_dark(CCDitem, image=None):
    """Subtracts the dark current from the image.

    Args:
        CCDitem:  dictonary containing CCD image and information
        image (optional) np.array: If this is given then it will be used instead of the image in the CCDitem

    Returns: 
        image_dark_sub (np.array, dtype=float64): true number of counts
        flags (np.array, dtype = uint16): 2 flags to indicate problems with the darc subtractions. 
            Binary array: 1st bit idicates that the dark subtraction renedered a negative value as result, second bit indiates a temperature out of normal range.
    """
    
    if image is None:
        image = CCDitem["IMAGE"]
        
    dark_fullpic = calculate_dark(CCDitem)

    
    image_dark_sub = (
        image
        - dark_fullpic[
            CCDitem["NRSKIP"] : CCDitem["NRSKIP"] + CCDitem["NROW"],
            CCDitem["NCSKIP"] : CCDitem["NCSKIP"] + CCDitem["NCOL"] + 1,
        ]
    )
    # rows,colums Note that nrow always seems to be implemented as +1 already, whereas NCOL does not, hence the missing '+1' in the column calculation /LM201204
    
    #If image becomes negative set flag
    error_flag_negative = np.zeros(image.shape, dtype=np.uint16)
    error_flag_negative[image_dark_sub<0] = 1
    error_flag_temperature=np.zeros(image.shape, dtype=np.uint16)
    if CCDitem["temperature"]<-50. or CCDitem["temperature"]>30.: # Filter out cases where the temperature seems wrong. 
        error_flag_temperature.fill(1)
    
    error_flag = combine_flags([make_binary(error_flag_negative,1), make_binary(error_flag_temperature,1)])

    return image_dark_sub, error_flag


def calculate_dark(CCDitem):
    try:
        CCDunit = CCDitem["CCDunit"]
    except:
        raise Exception("No CCDunit defined for the CCDitem")
    #    CCDunit=CCD(CCDitem['channel'])
    totdarkcurrent = (
        CCDunit.darkcurrent2D(CCDitem["temperature"], CCDitem["GAIN Mode"])
        * int(CCDitem["TEXPMS"])
        / 1000.0
    )  # tot dark current in electrons
    # totbinpix = int(CCDitem["NCBIN CCDColumns"]) * int(
    #    CCDitem["NCBIN FPGAColumns"]
    # ) 
    dark_calc_image = (
        CCDunit.ampcorrection
        * totdarkcurrent
        / CCDunit.alpha_avr(CCDitem["GAIN Mode"])
    )

    if (
        (CCDitem["NCBIN CCDColumns"] > 1)
        or (CCDitem["NCBIN FPGAColumns"] > 1)
        or (CCDitem["NRBIN"] > 1)
    ):  # Or row binning
        dark_calc_image = bin_image_using_predict_and_get_true_image(
            CCDitem, dark_calc_image
        )
    return dark_calc_image


def predict_image(reference_image, hsm_header, lsm_image, lsm_header, header):
    """
    this is a function to predict an image read out from the CCD with a given set
    of parameters, based on a reference image (of size 511x2048)
    """
    ncol = int(header["NCol"]) + 1
    nrow = int(header["NRow"])

    nrowskip = int(header["NRowSkip"])
    ncolskip = int(header["NColSkip"])

    nrowbin = int(header["NRowBinCCD"])
    ncolbinC = int(header["NCBIN CCDColumns"])
    ncolbinF = int(header["NCBIN FPGAColumns"])

    if int(header["SignalMode"]) > 0:
        blank = int(lsm_header["BlankTrailingValue"])
    else:
        blank = int(hsm_header["BlankTrailingValue"])

    blank_off = blank - 128
    zerolevel = int(header["ZeroLevel"])

    gain = 2 ** (int(header["Gain"]) & 255)

    bad_columns = header["BadCol"]
    print("bad", bad_columns)

    if nrowbin == 0:  # no binning means beaning of one
        nrowbin = 1

    if ncolbinC == 0:  # no binning means beaning of one
        ncolbinC = 1

    if ncolbinF == 0:  # no binning means beaning of one
        ncolbinF = 1

    ncolbintotal = ncolbinC * ncolbinF

    if int(header["SignalMode"]) > 0:
        reference_image = get_true_image(lsm_header, lsm_image)
        reference_image = desmear_true_image(lsm_header, reference_image)
    else:
        reference_image = get_true_image(hsm_header, reference_image)
        reference_image = desmear_true_image(hsm_header, reference_image)

    # bad column analysis
    n_read, n_coadd = binning_bc(ncol, ncolskip, ncolbinF, ncolbinC, header["BadCol"])

    image = np.zeros((nrow, ncol))
    image[:, :] = 128  # offset

    finished_row = 0
    finished_col = 0
    for j_r in range(0, nrow):  # check indexing again
        for j_c in range(0, ncol):
            for j_br in range(0, nrowbin):  # account for row binning on CCD
                if j_br == 0:
                    image[j_r, j_c] = (
                        image[j_r, j_c] + n_read[j_c] * blank_off
                    )  # here we add the blank value, only once per binned row
                    # (LM201025 n_read is the number a superbin has been devided into to be read. Why n_read when we are doing row binning. Shouldnt n_read always be one here? No, not if that supercolumn had a BC and was read out in two bits.
                for j_bc in range(0, ncolbintotal):  # account for column binning
                    # LM201030: Go through all unbinned columns(both from FPGA and onchip) that belongs to one superpixel(j_r,j_c) and if the column is not Bad, add the signal of that unbinned pixel to the superpixel (j_r,j_c)
                    # out of reference image range
                    if (j_r) * nrowbin + j_br + nrowskip > 511:
                        break
                    elif (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip > 2048:
                        break

                    # removed +1 after bad_columns, unclear why it was added
                    # TODO
                    #   print('mycolumn: ',(j_c)*ncolbinC*ncolbinF + j_bc + ncolskip)
                    #   print('j_br: ',j_br, ' j_c=', j_c, ' j_bc=', j_bc)
                    #                    if j_r==0 and j_br==0 and j_bc==0:
                    #                        print('start bin mycolumn=',(j_c)*ncolbinC*ncolbinF + j_bc + ncolskip)

                    if ncolbinC > 1 and (
                        j_c
                    ) * ncolbinC * ncolbinF + j_bc + ncolskip in (
                        bad_columns
                    ):  # KTH:should be here +1 becuase Ncol is +1
                        # 201106 LM removed +1. Checked by using picture: recv_image, recv_header = readimgpath('/Users/lindamegner/MATS/retrieval/Level1/data/2019-02-08 rand6/', 32, 0)

                        # if j_r==0 and j_br==0:
                        #     print('skipping BC mycolumn=',(j_c)*ncolbinC*ncolbinF + j_bc + ncolskip)
                        #     print('mycolumn: ',(j_c)*ncolbinC*ncolbinF + j_bc + ncolskip)
                        #     print('j_br: ',j_br, ' j_c=', j_c, ' j_bc=', j_bc)
                        # image[j_r, j_c] =-999
                        continue
                    else:

                        # add only the actual signal from every pixel (minus blank)
                        image[j_r, j_c] = (
                            image[j_r, j_c]  # remove blank
                            # LM201103 fixed bug renmoved -1 from th
                            # + reference_image[(j_r-1)*nrowbin+j_br+nrowskip-1,(j_c-1)*ncolbinC*ncolbinF+j_bc+ncolskip-1] #row and column value evaluation, -1 to adjust for python indexing
                            + reference_image[
                                (j_r) * nrowbin + j_br + nrowskip,
                                (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip,
                            ]  # row and column value evaluation
                            * 1  # scaling factor
                        )

    image = image / gain
    pred_header = header
    pred_header["BlankTrailingValue"] = blank

    return image, pred_header


def meanbin_image_with_BC(header, reference_image="999"):
    """
    this is a function to bin an image withouyt any offset or blanks. Bad columns are skipped.
    code is modified from Georgis predict_image
    """

    ncol = int(header["NCOL"]) + 1
    nrow = int(header["NROW"])

    nrowskip = int(header["NRSKIP"])
    ncolskip = int(header["NCSKIP"])

    nrowbin = int(header["NRBIN"])
    ncolbinC = int(header["NCBIN CCDColumns"])
    ncolbinF = int(header["NCBIN FPGAColumns"])

    bad_columns = header["BC"]

    if nrowbin == 0:  # no binning means beaning of one
        nrowbin = 1

    if ncolbinC == 0:  # no binning means beaning of one
        ncolbinC = 1

    if ncolbinF == 0:  # no binning means beaning of one
        ncolbinF = 1

    ncolbintotal = ncolbinC * ncolbinF

    if reference_image == "999":
        reference_image = header["IMAGE"]

    # bad column analysis
    #   n_read, n_coadd = binning_bc(ncol, ncolskip, ncolbinF, ncolbinC, header['BC'])

    image = np.zeros((nrow, ncol))  # no offset
    nr_of_entries = np.zeros((nrow, ncol))

    for j_r in range(0, nrow):  # check indexing again
        for j_c in range(0, ncol):
            for j_br in range(0, nrowbin):  # account for row binning on CCD
                for j_bc in range(0, ncolbintotal):  # account for column binning
                    # LM201030: Go through all unbinned columns(both from FPGA and onchip) that belongs to one superpixel(j_r,j_c) and if the column is not Bad, add the signal of that unbinned pixel to the superpixel (j_r,j_c)
                    # out of reference image range
                    if (j_r) * nrowbin + j_br + nrowskip > 511:
                        break
                    elif (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip > 2048:
                        break

                    # removed +1 after bad_columns, unclear why it was added
                    # TODO
                    if (
                        ncolbinC > 1
                        and (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip in bad_columns
                    ):  # +1 becuase Ncol is +1
                        continue
                    else:

                        # add only the actual signal from every pixel (minus blank)
                        image[j_r, j_c] = (
                            image[j_r, j_c]  # remove blank
                            # LM201103 fixed bug renmoved -1 from th
                            # + reference_image[(j_r-1)*nrowbin+j_br+nrowskip-1,(j_c-1)*ncolbinC*ncolbinF+j_bc+ncolskip-1] #row and column value evaluation, -1 to adjust for python indexing
                            + reference_image[
                                (j_r) * nrowbin + j_br + nrowskip,
                                (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip,
                            ]  # row and column value evaluation
                            * 1  # scaling factor
                        )

                        nr_of_entries[j_r, j_c] = nr_of_entries[j_r, j_c] + 1

    mean_image = image / nr_of_entries
    return mean_image


def bin_image_with_BC(header, reference_image="999"):
    """
    this is a function to bin an image withouyt any offset or blanks. Bad columns are skipped.
    code is modified from Georgis predict_image
    """

    ncol = int(header["NCOL"]) + 1
    nrow = int(header["NROW"])

    nrowskip = int(header["NRSKIP"])
    ncolskip = int(header["NCSKIP"])

    nrowbin = int(header["NRBIN"])
    ncolbinC = int(header["NCBIN CCDColumns"])
    ncolbinF = int(header["NCBIN FPGAColumns"])

    bad_columns = header["BC"]

    if nrowbin == 0:  # no binning means beaning of one
        nrowbin = 1

    if ncolbinC == 0:  # no binning means beaning of one
        ncolbinC = 1

    if ncolbinF == 0:  # no binning means beaning of one
        ncolbinF = 1

    ncolbintotal = ncolbinC * ncolbinF

    if reference_image == "999":
        reference_image = header["IMAGE"]

    # bad column analysis
    #   n_read, n_coadd = binning_bc(ncol, ncolskip, ncolbinF, ncolbinC, header['BC'])

    image = np.zeros((nrow, ncol))  # no offset
    nr_of_entries = np.zeros((nrow, ncol))

    for j_r in range(0, nrow):  # check indexing again
        for j_c in range(0, ncol):
            for j_br in range(0, nrowbin):  # account for row binning on CCD
                for j_bc in range(0, ncolbintotal):  # account for column binning
                    # LM201030: Go through all unbinned columns(both from FPGA and onchip) that belongs to one superpixel(j_r,j_c) and if the column is not Bad, add the signal of that unbinned pixel to the superpixel (j_r,j_c)
                    # out of reference image range
                    if (j_r) * nrowbin + j_br + nrowskip > 511:
                        break
                    elif (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip > 2048:
                        break

                    # removed +1 after bad_columns, unclear why it was added
                    # TODO
                    if (
                        ncolbinC > 1
                        and (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip in bad_columns
                    ):  # +1 becuase Ncol is +1
                        continue
                    else:

                        # add only the actual signal from every pixel (minus blank)
                        image[j_r, j_c] = (
                            image[j_r, j_c]  # remove blank
                            # LM201103 fixed bug renmoved -1 from th
                            # + reference_image[(j_r-1)*nrowbin+j_br+nrowskip-1,(j_c-1)*ncolbinC*ncolbinF+j_bc+ncolskip-1] #row and column value evaluation, -1 to adjust for python indexing
                            + reference_image[
                                (j_r) * nrowbin + j_br + nrowskip,
                                (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip,
                            ]  # row and column value evaluation
                            * 1  # scaling factor
                        )

                        # nr_of_entries[j_r, j_c] = nr_of_entries[j_r, j_c]+1

    # mean_image=image/nr_of_entries
    return image


def bin_image_using_predict(header, reference_image="999"):
    """
    this is a function to predict an image read out from the CCD with a given set
    of parameters, based on a reference image (of size 511x2048)
    """

    ncol = int(header["NCOL"]) + 1
    nrow = int(header["NROW"])

    nrowskip = int(header["NRSKIP"])
    ncolskip = int(header["NCSKIP"])

    nrowbin = int(header["NRBIN"])
    ncolbinC = int(header["NCBIN CCDColumns"])
    ncolbinF = int(header["NCBIN FPGAColumns"])

    blank = int(header["TBLNK"])

    blank_off = blank - 128

    # gain=2**(int(header['Gain']) & 255) #use for old data format
    gain = 2.0 ** header["GAIN Truncation"]
    bad_columns = header["BC"]

    if nrowbin == 0:  # no binning means beaning of one
        nrowbin = 1

    if ncolbinC == 0:  # no binning means beaning of one
        ncolbinC = 1

    if ncolbinF == 0:  # no binning means beaning of one
        ncolbinF = 1

    ncolbintotal = ncolbinC * ncolbinF

    if type(reference_image) == str:
        if reference_image == "999":
            reference_image = header["IMAGE"]
        else:
            raise ValueError('Invalid reference image')

    # bad column analysis
    n_read, n_coadd = binning_bc(ncol, ncolskip, ncolbinF, ncolbinC, header["BC"])

    image = np.zeros((nrow, ncol))
    image[:, :] = 128  # offset

    finished_row = 0
    finished_col = 0
    for j_r in range(0, nrow):  # check indexing again
        for j_c in range(0, ncol):
            for j_br in range(0, nrowbin):  # account for row binning on CCD
                if j_br == 0:
                    image[j_r, j_c] = (
                        image[j_r, j_c] + n_read[j_c] * blank_off
                    )  # here we add the blank value, only once per binned row
                    # (LM201025 n_read is the number a superbin has been devided into to be read. So if no badcolums or fpga binning then n_read=1.
                for j_bc in range(0, ncolbintotal):  # account for column binning
                    # LM201030: Go through all unbinned columns(both from FPGA and onchip) that belongs to one superpixel(j_r,j_c) and if the column is not Bad, add the signal of that unbinned pixel to the superpixel (j_r,j_c)
                    # out of reference image range
                    if (j_r) * nrowbin + j_br + nrowskip > 511:
                        break
                    elif (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip > 2048:
                        break

                    # removed +1 after bad_columns, unclear why it was added
                    # TODO
                    if (
                        ncolbinC > 1
                        and (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip in bad_columns
                    ):  # +1 becuase Ncol is +1
                        continue
                    else:

                        # add only the actual signal from every pixel (minus blank)
                        image[j_r, j_c] = (
                            image[j_r, j_c]  # remove blank
                            # LM201103 fixed bug renmoved -1 from th
                            # + reference_image[(j_r-1)*nrowbin+j_br+nrowskip-1,(j_c-1)*ncolbinC*ncolbinF+j_bc+ncolskip-1] #row and column value evaluation, -1 to adjust for python indexing
                            + reference_image[
                                (j_r) * nrowbin + j_br + nrowskip,
                                (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip,
                            ]  # row and column value evaluation
                        )

    binned_image = image / gain

    return binned_image, header


def bin_image_using_predict_and_get_true_image(header, reference_image="999"):
    simage_raw_binned, header = bin_image_using_predict(header, reference_image)
    simage_raw_binned, _ = get_true_image(header, simage_raw_binned)
    return simage_raw_binned


def winmode_correct(CCDitem):
    if (CCDitem["WinMode"]) <= 4:
        winfactor = 2 ** CCDitem["WinMode"]
    elif (CCDitem["WinMode"]) == 7:
        winfactor = 1
    else:
        raise Exception("Undefined Window")
    image_lsb = winfactor * CCDitem["IMAGE"]
    return image_lsb


def get_true_image_opposite_order(image, header):
    true_image = get_true_image(header, image)
    return true_image


def get_true_image(header, image=None):
    # calculate true image by removing readout offset (pixel blank value) and
    # compensate for bad colums

    if image is None:  
        image = header["IMAGE"]

    # Both 0 and 1 means that no binning has been done on CCD (depricated)
    ncolbinC = int(header["NCBIN CCDColumns"])
    if ncolbinC == 0:
        ncolbinC = 1

    # correct for digital gain from FPGA binning
    true_image = image * 2 ** (
        int(header["GAIN Truncation"])
    )  # Says Gain in original coding.  Check with Nickolay LM 201215
    # Check if this is same as "GAIN Truckation". Check with Molflow if this is corrected for already in  Level 0. 
     

    # bad column analysis #LM201025 nread appears to be number of bins or (super)columns binned in FPGA (and bad columns), coadd is numer of total columns in a supersupercolumn (FPGA and onchip binned)
    n_read, n_coadd = binning_bc(
        int(header["NCOL"]) + 1,
        int(header["NCSKIP"]),
        int(header["NCBIN FPGAColumns"]),
        ncolbinC,
        header["BC"],
    )

    # go through the columns
    for j_c in range(0, int(header["NCOL"] + 1)):  # LM201102 Big fix +1
        # remove blank values and readout offsets
        bc_comp_fact=(int(header["NCBIN FPGAColumns"]) * ncolbinC / n_coadd[j_c])
        true_image[0 : int(header["NROW"]) + 1, j_c] = (
            true_image[0 : int(header["NROW"] + 1), j_c]
            - (n_read[j_c] * (header["TBLNK"] - 128)
            + 128)*  bc_comp_fact
        )

    #If image becomes negative set flag
    error_flag = np.zeros(true_image.shape, dtype=np.uint16)
    error_flag[true_image<0] = 1
    
    error_flag = make_binary(error_flag,1)

    return true_image, error_flag


def get_true_image_reverse(header, true_image=None):
    # add readout offset (pixel blank value) and bad colums stuff,
    # by reversing get_true_image

    if true_image is None:  
        true_image = header["IMAGE"]

    ncolbinC = int(header["NCBIN CCDColumns"])
    if ncolbinC == 0:
        ncolbinC = 1

    # bad column analysis
    n_read, n_coadd = binning_bc(
        int(header["NCOL"]) + 1,
        int(header["NCSKIP"]),
        int(header["NCBIN FPGAColumns"]),
        ncolbinC,
        header["BC"],
    )

    # go through the columns
    for j_c in range(0, int(header["NCOL"]) + 1):
        # compensate for bad columns
        true_image[0 : int(header["NROW"]) + 1, j_c] = true_image[
            0 : int(header["NROW"] + 1), j_c
        ] / (int(header["NCBIN FPGAColumns"]) * ncolbinC / n_coadd[j_c])

        # add blank values and readout offsets
        true_image[0 : int(header["NROW"]) + 1, j_c] = (
            true_image[0 : int(header["NROW"] + 1), j_c]
            + n_read[j_c] * (header["TBLNK"] - 128)
            + 128
        )

    # add gain
    image = true_image / 2 ** (
        int(header["GAIN Truncation"])
    )  # Check with Molflow if this is corrected for already in  Level 0. 
    return image


def get_true_image_from_compensated(image, header):

    # calculate true image by removing readout offset, pixel blank value and
    # normalising the signal level according to readout time

    # remove gain
    true_image = image * 2 ** (int(header["Gain"]) & 255)

    for j_c in range(0, int(header["NCol"]) + 1):  # LM201102 Big fix +1 added
        true_image[0 : header["NRow"], j_c] = (
            true_image[0 : header["NRow"], j_c]
            - header["NCBIN FPGAColumns"] * (header["BlankTrailingValue"] - 128)
            - 128
        )

    return true_image


def binning_bc(Ncol, Ncolskip, NcolbinFPGA, NcolbinCCD, BadColumns):

    # a routine to estimate the correction factors for column binning with bad columns

    # n_read - array, containing the number of individually read superpixels
    #           attributing to the given superpixel
    # n_coadd - array, containing the number of co-added individual pixels
    # Input - as per ICD. BadColumns - array containing the index of bad columns
    #           (the index of first column is 0)

    n_read = np.zeros(Ncol)
    n_coadd = np.zeros(Ncol)

    col_index = Ncolskip

    for j_col in range(0, Ncol):
        for j_FPGA in range(0, NcolbinFPGA):
            continuous = 0
            for j_CCD in range(0, NcolbinCCD):
                if col_index in BadColumns:
                    if continuous == 1:
                        n_read[j_col] = n_read[j_col] + 1
                    continuous = 0
                else:
                    continuous = 1
                    n_coadd[j_col] = n_coadd[j_col] + 1

                col_index = col_index + 1

            if continuous == 1:
                n_read[j_col] = n_read[j_col] + 1

    return n_read, n_coadd



def desmear_true_image(header, image=None):
    """Subtracts the smearing (due to no shutter) from the image.

    Args:
        CCDitem:  dictonary containing CCD image and information
        image (optional) np.array: If this is given then it will be used instead of the image in the CCDitem

    Returns: 
        image (np.array, dtype=float64): desmeared image
        flag (np.array, dtype = uint16): error flag to indicate that the de-smearing gave a negative value as a reslut    
        """
    
    if image is None:
        image = header["IMAGE"]

    nrow = int(header["NROW"])
    ncol = int(header["NCOL"]) + 1
    # calculate extra time per row
    T_row_extra, T_delay = calculate_time_per_row(header)
    T_exposure = float(header["TEXPMS"]) / 1000.0

    TotTime = 0
    for irow in range(1, nrow):
        for krow in range(0, irow):
            image[irow, 0:ncol] = image[irow, 0:ncol] - image[krow, 0:ncol] * (
                T_row_extra / T_exposure
            )
            TotTime = TotTime + T_row_extra

    # row 0 here is the first row to read out from the chip
    
    # Flag for negative values
    error_flag= np.zeros(image.shape, dtype=np.uint16)
    error_flag[image<0] = 1

    error_flag = make_binary(error_flag,2)

    return image, error_flag


def desmear_true_image_reverse(header, image=None):
    # add readout offset (pixel blank value) and bad colums stuff,
    # by reversing get_true_image

    if image is None:
        image = header["IMAGE"]

    nrow = int(header["NROW"])
    ncol = int(header["NCOL"]) + 1
    # calculate extra time per row
    T_row_extra, T_delay = calculate_time_per_row(header)
    T_exposure = (
        float(header["TEXPMS"]) / 1000.0
    )  # check for results when shifting from python 2 to 3

    TotTime = 0
    # row 0 here is the first row to read out from the chip
    # # #Code version 1 (copy the nonsmeared image) to desmear, the result should be the same as code version 1
    # tmpimage=image.copy()
    # for irow in range(1,nrow):
    #     for krow in range(0,irow):
    #         image[irow,0:ncol]=image[irow,0:ncol] + tmpimage[krow,0:ncol]*((T_row_extra)/T_exposure)
    #         TotTime=TotTime+T_row_extra

    # Code version 2 (loop backwards)to desmear, the result should be the same as code version 2
    for irow in range(nrow - 1, 0, -1):
        for krow in range(0, irow):
            image[irow, 0:ncol] = image[irow, 0:ncol] + image[krow, 0:ncol] * (
                T_row_extra / T_exposure
            )
            TotTime = TotTime + T_row_extra

    return image


def calculate_time_per_row(header):

    # this function provides some useful timing data for the CCD readout

    # Note that minor "transition" states may have been omitted resulting in
    # somewhat shorter readout times (<0.1%).

    # Default timing setting is_
    # ccd_r_timing <= x"A4030206141D"&x"010303090313"

    # All pixel timing setting is the final count of a counter that starts at 0,
    # so the number of clock cycles exceeds the setting by 1

    # image parameters
    ncol = int(header["NCOL"]) + 1
    ncolbinC = int(header["NCBIN CCDColumns"])
    if ncolbinC == 0:
        ncolbinC = 1
    ncolbinF = int(header["NCBIN FPGAColumns"])

    nrow = int(header["NROW"])
    nrowbin = int(header["NRBIN"])
    if nrowbin == 0:
        nrowbin = 1
    nrowskip = int(header["NRSKIP"])

    n_flush = int(header["NFLUSH"])

    # timing settings
    full_timing = 1  # TODO <-- meaning?

    # full pixel readout timing n#TODO discuss this with OM,  LM these are default values. change these when the header contians this infromation

    time0 = 1 + 19  # x13%TODO
    time1 = 1 + 3  # x03%TODO
    time2 = 1 + 9  # x09%TODO
    time3 = 1 + 3  # x03%TODO
    time4 = 1 + 3  # x03%TODO
    time_ovl = 1 + 1  # x01%TODO

    # fast pixel readout timing
    timefast = 1 + 2  # x02%TODO
    timefastr = 1 + 3  # x03%TODO

    # row shift timing
    row_step = 1 + 164  # xA4%TODO

    clock_period = 30.517  # master clock period, ns 32.768 MHz

    # there is one extra clock cycle, effectively adding to time 0
    Time_pixel_full = (
        1 + time0 + time1 + time2 + time3 + time4 + 3 * time_ovl
    ) * clock_period

    # this is the fast timing pixel period
    Time_pixel_fast = (1 + 4 * timefast + 3 * time_ovl + timefastr) * clock_period

    # here we calculate the number of fast and slow pixels
    # NOTE: the effect of bad pixels is disregarded here

    if full_timing == 1:
        n_pixels_full = 2148
        n_pixels_fast = 0
    else:
        if ncolbinC < 2:  # no CCD binning
            n_pixels_full = ncol * ncolbinF
        else:  # there are two "slow" pixels for one superpixel to be read out
            n_pixels_full = 2 * ncol * ncolbinF
        n_pixels_fast = 2148 - n_pixels_full

    # time to read out one row
    T_row_read = n_pixels_full * Time_pixel_full + n_pixels_fast * Time_pixel_fast

    # shift time of a single row
    T_row_shift = (64 + row_step * 10) * clock_period

    # time of the exposure start delay from the start_exp signal # n_flush=1023
    T_delay = T_row_shift * n_flush

    # total time of the readout
    T_readout = T_row_read * (nrow + nrowskip + 1) + T_row_shift * (1 + nrowbin * nrow)

    # "smearing time"
    # (this is the time that any pixel collects electrons in a wrong row, during the shifting.)
    # For smearing correction, this is the "extra exposure time" for each of the rows.

    T_row_extra = (T_row_read + T_row_shift * nrowbin) / 1e9

    return T_row_extra, T_delay


def compare_image(image1, image2, header):

    # this is a function to compare two images of the same size
    # one comparison is a linear fit of columns, the other comparison is a linear fit
    # of rows, the third is a linear fit of the whole image

    sz1 = image1.shape
    sz2 = image2.shape

    if sz1[0] != sz2[0] or sz1[1] != sz2[1]:
        print("sizes of input images do not match")

    nrow = sz1[0]
    ncol = sz1[1]

    nrowskip = int(header["NRSKIP"])
    ncolskip = int(header["NCSKIP"])

    nrowbin = int(header["NRBIN"])
    ncolbinC = int(header["NCBIN"])
    ncolbinF = int(header["NCBIN FPGAColumns"])

    if nrowskip + nrowbin * nrow > 511:
        nrow = np.floor((511 - nrowskip) / nrowbin)

    if ncolskip + ncolbinC * ncolbinF * ncol > 2047:
        nrow = np.floor((2047 - ncolskip) / (ncolbinC * ncolbinF))
    print(nrow, image1.shape)
    image1 = image1[0 : nrow - 1, 0 : ncol - 1]
    image2 = image2[0 : nrow - 1, 0 : ncol - 1]

    r_scl = np.zeros(nrow)
    r_off = np.zeros(nrow)
    r_std = np.zeros(nrow)

    for jj in range(0, nrow - 1):
        x = np.concatenate(
            (
                np.ones((ncol - 1, 1)),
                np.expand_dims(
                    image1[
                        jj,
                    ]
                    .conj()
                    .transpose(),
                    axis=1,
                ),
            ),
            axis=1,
        )  # -1 to adjust to python indexing?
        y = (
            image2[
                jj,
            ]
            .conj()
            .transpose()
        )
        bb, ab, aa, cc = np.linalg.lstsq(x, y)

        ft = np.squeeze([a * bb[1] for a in x[:, 1]]) + bb[0]
        # ft=np.multiply(x[:,1]*bb[1]) + bb[0]

        adf = np.abs(np.squeeze(y) - np.squeeze(ft))
        sigma = np.std(np.squeeze(y) - np.squeeze(ft))

        inside = np.where(adf < 2 * sigma)
        bb, ab, aa, cc = np.linalg.lstsq(
            x[
                inside[1],
            ],
            y[inside[1]],
        )

        ft = np.squeeze([a * bb[1] for a in x[:, 1]]) + bb[0]

        r_scl[jj] = bb[1]
        r_off[jj] = bb[0]
        r_std[jj] = np.std(y[0] - ft[0])

    c_scl = np.zeros(nrow)
    c_off = np.zeros(nrow)
    c_std = np.zeros(nrow)

    for jj in range(0, ncol - 1):

        x = np.concatenate(
            (np.ones((nrow - 1, 1)), np.expand_dims(image1[:, jj], axis=1)), axis=1
        )
        y = image2[:, jj]
        bb, ab, aa, cc = np.linalg.lstsq(x, y)

        ft = np.squeeze([a * bb[1] for a in x[:, 1]]) + bb[0]

        adf = np.abs(np.squeeze(y) - np.squeeze(ft))
        sigma = np.std(np.squeeze(y) - np.squeeze(ft))

        inside = np.where(adf < 2 * sigma)
        bb, ab, aa, cc = np.linalg.lstsq(
            x[
                inside[1],
            ],
            y[inside[1]],
        )

        ft = np.squeeze([a * bb[1] for a in x[:, 1]]) + bb[0]

        c_scl[jj] = bb[1]
        c_off[jj] = bb[0]
        c_std[jj] = np.std(y[0] - ft[0])

    nsz = (nrow - 1) * (ncol - 1)
    la_1 = np.reshape(image1, (nsz, 1))
    la_2 = np.reshape(image2, (nsz, 1))

    x = np.concatenate((np.ones((nsz, 1)), la_1), axis=1)
    y = la_2
    bb, ab, aa, cc = np.linalg.lstsq(x, y)

    ft = np.squeeze([a * bb[1] for a in x[:, 1]]) + bb[0]

    adf = np.abs(np.squeeze(y) - np.squeeze(ft))
    sigma = np.std(np.squeeze(y) - np.squeeze(ft))

    inside = np.where(adf < 2 * sigma)
    bb, ab, aa, cc = np.linalg.lstsq(
        x[
            inside[1],
        ],
        y[inside[1]],
    )

    ft = np.squeeze([a * bb[1] for a in x[:, 1]]) + bb[0]

    t_off = bb[0]
    t_scl = bb[1]
    t_std = np.std(y[0] - ft[0])

    rows = 0

    return t_off, t_scl, t_std


def compensate_bad_columns(header, image="No picture"):

    if type(image) is str:
        image = header["IMAGE"].copy()

    # LM 200127 This does not need to be used since it is already done in the OBC says Georgi.

    # this is a function to compensate bad columns if in the image

    ncol = int(header["NCOL"]) + 1
    nrow = int(header["NROW"])

    ncolskip = int(header["NCSKIP"])

    ncolbinC = int(header["NCBIN CCDColumns"])
    ncolbinF = int(header["NCBIN FPGAColumns"])

    # change to Leading if Trailing does not work properly
    blank = int(header["TBLNK"])

    gain = 2 ** (
        int(header["GAIN Truncation"])
    )  
    if ncolbinC == 0:  # no binning means binning of one
        ncolbinC = 1

    if ncolbinF == 0:  # no binning means binning of one
        ncolbinF = 1

    # bad column analysis

    n_read, n_coadd = binning_bc(
        ncol, ncolskip, ncolbinF, ncolbinC, np.asarray(header["BC"])
    )

    if ncolbinC > 1:
        for j_c in range(0, ncol):
            if ncolbinC * ncolbinF != n_coadd[j_c]:
                # remove gain adjustment
                image[0 : nrow - 1, j_c] = image[0 : nrow - 1, j_c] * gain

                # remove added superpixel value due to bad columns and read out offset
                image[0 : nrow - 1, j_c] = (
                    image[0 : nrow - 1, j_c] - n_read[j_c] * (blank - 128) - 128
                )

                # multiply by number of binned column to actual number readout ratio
                image[0 : nrow - 1, j_c] = image[0 : nrow - 1, j_c] * (
                    (ncolbinC * ncolbinF) / n_coadd[j_c]
                )

                # add number of FPGA binned
                image[0 : nrow - 1, j_c] = (
                    image[0 : nrow - 1, j_c] + ncolbinF * (blank - 128) + 128
                )

                # add gain adjustment back
                image[0 : nrow - 1, j_c] = image[0 : nrow - 1, j_c] / gain

                # print('Col: ',j_c,', n_read: ',n_read[j_c],', n_coadd: ',n_coadd[j_c],', binned pixels: ',ncolbinC*ncolbinF)

    return image


#
# def get_true_image_from_compensated(image, header):
#
#    #calculate true image by removing readout offset, pixel blank value and
#    #normalising the signal level according to readout time
#
#    #remove gain
#    true_image = image * 2**(int(header['DigGain']) & 255)
#
#    for j_c in range(0,int(header['NCol'])):
#        true_image[0:header['NRow'], j_c] = ( true_image[0:header['NRow'],j_c] -
#                  header['NCBIN FPGAColumns'] * (header['BlankTrailingValue']-128) - 128 )
#
#
#    return true_image


def read_flatfield(CCDunit, mode, flatfield_directory):
    from mats_l1_processing.items_units_functions import (
        read_files_in_protocol_as_ItemsUnits,
    )
    from mats_l1_processing.experimental_utils import readprotocol

    # Note that 1 and 0 are switched  for signal mode

    if mode == 0:  # HSM
        directory = flatfield_directory
        # protocol='flatfields_200330_SigMod1_LMprotocol.txt'
        protocol = "readin_flatfields_SigMod1.txt"

    elif mode == 1:  # LSM
        directory = flatfield_directory

        protocol = "readin_flatfields_SigMod0.txt"
    else:
        print("Undefined mode")

    read_from = "rac"
    df_protocol = readprotocol(directory + protocol)
    # df_only2 = df_protocol[(df_protocol.index-2) % 3 != 0]

    # The below reads all images in protocol - very inefficient. Should be only one file read in LM200810
    CCDItemsUnits = read_files_in_protocol_as_ItemsUnits(
        df_protocol, directory, 3, read_from
    )
    # Pick the rignt image, thsi should be hard coded in the end

    if CCDunit.channel == "NADIR":  # Hack since we dont have any nadir flat fields yet.
        # Cannot be zero due to zero devision in calculate_flatfield. Should be fixed.
        flatfield = np.zeros((511, 2048)) + 0.01

    else:
        CCDItemsUnitsSelect = list(
            filter(lambda x: (x.imageItem["channel"] == CCDunit.channel), CCDItemsUnits)
        )

        if len(CCDItemsUnitsSelect) > 1:
            print("Several possible pictures found")
        try:
            flatfield = CCDItemsUnitsSelect[
                0
            ].subpic  # This is where it gets read in. The dark (including offsets and balnks) have already been subracted.
        except:
            print("No flatfield CCDItemUnit found - undefined flatfield")

    return flatfield


def readimg(filename):

    data_arr = np.fromfile(filename, dtype="uint16")
    # convert header to binary
    header_bin = np.asarray(
        [bin(data_arr[i]) for i in range(0, 12)]
    )  # this is a string
    # change format of header_bin elements to be formatted like matlab char array
    # print(header_bin)
    for i in range(0, len(header_bin)):
        header_bin[i] = header_bin[i][2:].zfill(16)
    # print(header_bin)
    # read header
    Frame_count = int(header_bin[0], 2)
    NRow = int(header_bin[1], 2)
    NRowBinCCD = int(header_bin[2][10:16], 2)
    NRowSkip = int(header_bin[3][8:16], 2)
    NCol = int(header_bin[4], 2)
    NColBinFPGA = int(header_bin[5][2:8], 2)
    NColBinCCD = int(header_bin[5][8:16], 2)
    NColSkip = int(header_bin[6][5:16], 2)
    N_flush = int(header_bin[7], 2)
    Texposure_MSB = int(header_bin[8], 2)
    Texposure_LSB = int(header_bin[9], 2)
    Gain = int(header_bin[10], 2)
    SignalMode = Gain & 4096
    Temperature_read = int(header_bin[11], 2)
    # print(len(header_bin[4]))
    # read image
    if len(data_arr) < NRow * (NCol + 1) / (
        2 ** (NColBinFPGA)
    ):  # check for differences in python 2 and 3
        img_flag = 0
        image = 0
        Noverflow = 0
        BlankLeadingValue = 0
        BlankTrailingValue = 0
        ZeroLevel = 0

        Reserved1 = 0
        Reserved2 = 0

        Version = 0
        VersionDate = 0
        NBadCol = 0
        BadCol = 0
        Ending = "Wrong size"
    else:
        img_flag = 1
        image = np.reshape(
            np.double(data_arr[11 + 1 : NRow * (NCol + 1) + 12]), (NRow, NCol + 1)
        )  # LM201102 Corrected bug where4 NCol and NRow were switched
        # image = np.matrix(image).getH()

        # Trailer
        trailer_bin = np.asarray(
            [bin(data_arr[i]) for i in range(NRow * (NCol + 1) + 12, len(data_arr))]
        )
        for i in range(0, len(trailer_bin)):
            trailer_bin[i] = trailer_bin[i][2:].zfill(16)
        Noverflow = int(trailer_bin[0], 2)
        BlankLeadingValue = int(trailer_bin[1], 2)
        BlankTrailingValue = int(trailer_bin[2], 2)
        ZeroLevel = int(trailer_bin[3], 2)

        Reserved1 = int(trailer_bin[4], 2)
        Reserved2 = int(trailer_bin[5], 2)

        Version = int(trailer_bin[6], 2)
        VersionDate = int(trailer_bin[7], 2)
        NBadCol = int(trailer_bin[8], 2)
        BadCol = []
        Ending = int(trailer_bin[-1], 2)

        if NBadCol > 0:
            BadCol = np.zeros(NBadCol)
            for k_bc in range(0, NBadCol):
                BadCol[k_bc] = int(
                    trailer_bin[9 + k_bc], 2
                )  # check if this actually works

    # for yet unknown reasons the header entries are in some cases stripped by the last few digits
    # this causes an incorrect conversion to decimal values (since part of the binary string is missing)
    # as of 2019-08-22 this behaviour could be reproduced on a different machine, but not explained
    # checking for the correct size of the binary strings avoids issues arising from wrongly converted data

    if len(header_bin[1]) < 16:
        raise Exception("Binary string shortened")

    # original matlab code uses structured array, as of 20-03-2019 implementation as dictionary seems to be more useful choice
    # decision might depend on further (computational) use of data, which is so far unknown to me
    header = {}
    header["Size"] = len(data_arr)
    header["Frame_count"] = Frame_count
    header["NRow"] = NRow
    header["NRowBinCCD"] = NRowBinCCD
    header["NRowSkip"] = NRowSkip
    header["NCol"] = NCol
    header["NCBIN FPGAColumns"] = 2**NColBinFPGA
    header["NCBIN CCDColumns"] = NColBinCCD
    header["NColSkip"] = NColSkip
    header["N_flush"] = N_flush
    header["Texposure"] = Texposure_LSB + Texposure_MSB * 2 ** 16
    header["Gain"] = Gain & 255
    header["SignalMode"] = SignalMode
    header["Temperature_read"] = Temperature_read
    header["Noverflow"] = Noverflow
    header["BlankLeadingValue"] = BlankLeadingValue
    header["BlankTrailingValue"] = BlankTrailingValue
    header["ZeroLevel"] = ZeroLevel
    header["Reserved1"] = Reserved1
    header["Reserved2"] = Reserved2
    header["Version"] = Version
    header["VersionDate"] = VersionDate
    header["NBadCol"] = NBadCol
    header["BadCol"] = BadCol
    header["Ending"] = Ending

    return image, header, img_flag


def readimgpath(path, file_number, plot):
    import time
    import matplotlib.pyplot as plt

    filename = "%sF_0%02d/D_0%04d" % (path, np.floor(file_number / 100), file_number)
    # when using Microsoft Windows as Operating system replace / in the string with \\
    image, header, img_flag = readimg(filename)

    if plot > 0:
        # do the plotting
        plt.figure()
        mean_img = np.mean(image)

        plt.imshow(
            image, cmap="jet", vmin=mean_img - 100, vmax=mean_img + 100, aspect="auto"
        )
        plt.title("CCD image")
        plt.xlabel("Pixels")
        plt.ylabel("Pixels")
        plt.show()  # equivalent to hold off

        print(header)
        time.sleep(0.1)

    return image, header
