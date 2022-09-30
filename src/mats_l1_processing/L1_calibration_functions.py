# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:09:57 2019

@author: franzk, Linda Megner
(Linda has removed some fuctions and added more . Original franzk script is L1_functions.py)


Functions used for MATS L1 processing, based on corresponding MATLAB scripts provided by Georgi Olentsenko and Mykola Ivchenko
The MATLAB script can be found here: https://github.com/OleMartinChristensen/MATS-image-analysis



"""

import numpy as np
import scipy.optimize as opt
from mats_l1_processing.instrument import nonLinearity as NL
from joblib import Parallel, delayed
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
        CCDitem['flipped'] = True

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
    """ Linearizes the image. At the moment not done for NADIR.

    Args:
        CCDitem:  dictonary containing CCD image and information
        image: np.array The image that will be linearised
    Returns: 
        image_linear (np.array, dtype=float64): linearised number of counts
        flag (np.array, dtype = uint16): flag to indicate problems with the linearisation
    """        
    image_linear = np.zeros(image_bias_sub.shape)
    error_flag = np.zeros(image_bias_sub.shape, dtype=np.uint16)
    
    if CCDitem["channel"]=='NADIR': # No linearisation of NADIR at the moment
        image_linear =image_bias_sub
    else:
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
    """Calibrates the image for eacvh pixel, ie, absolute relativ calibration and flatfield compensation 

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

   
    image_calib_nonflipped = (
        image/CCDitem["CCDunit"].calib_denominator(CCDitem["GAIN Mode"])
        / image_flatf_fact
    )
    # rows,colums Note that nrow always seems to be implemented as +1 already, whereas NCOL does not, hence the missing '+1' in the column calculation /LM201204


    error_flag= np.zeros(image.shape, dtype=np.uint16)
    error_flag[image_calib_nonflipped<0] = 1 # Flag for negative value
    
    error_flag = make_binary(error_flag,2)

    return image_calib_nonflipped, error_flag


def calculate_flatfield(CCDitem):
    """Calculates the flatfield factor for binned images, otherwise simply returns 
    the flatfield of the channel. This factor should be diveded with to comensate for flatfield.

    Args:
        CCDitem:  dictonary containing CCD image and information


    Returns: 
        image_flatf: np.array of the same size as the binned image, with factors 
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
        (CCDitem["NCSKIP"] > 1)
        or (CCDitem["NCSKIP"] > 1)
        or (CCDitem["NCBIN CCDColumns"] > 1)
        or (CCDitem["NCBIN FPGAColumns"] > 1)
        or (CCDitem["NRBIN"] > 1)
        ):  
        image_flatf = meanbin_image_with_BC(CCDitem, image_flatf)

    return image_flatf


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
        
    dark_img = calculate_dark(CCDitem)


    image_dark_sub = image-dark_img
        
 
    #If image becomes negative set flag
    error_flag_negative = np.zeros(image.shape, dtype=np.uint16)
    error_flag_negative[image_dark_sub<0] = 1
    error_flag_temperature=np.zeros(image.shape, dtype=np.uint16)
    if CCDitem["temperature"]<-50. or CCDitem["temperature"]>30.: # Filter out cases where the temperature seems wrong. 
        error_flag_temperature.fill(1)
    
    error_flag = combine_flags([make_binary(error_flag_negative,1), make_binary(error_flag_temperature,1)])

    return image_dark_sub, error_flag


def calculate_dark(CCDitem):
    """
    Calculates dark image from Gabriels measurements. The function reads 
    gabriels dark current images using temperature and gain mode as an input. 
    The function converts from electrons to counts and returns a correctly
    binned dark current image in unit counts.  

    Args:
        CCDitem:  dictonary containing CCD image and information

    Returns: 
        dark_calc_image: Full frame dark current image for given CCD item. 
    """

    try:
        CCDunit = CCDitem["CCDunit"]
    except:
        raise Exception("No CCDunit defined for the CCDitem")
    #    CCDunit=CCD(CCDitem['channel'])
    
    #First estimate dark current using 0D algorithm
    totdarkcurrent= (CCDunit.darkcurrent(CCDitem["temperature"], CCDitem["GAIN Mode"]) 
        * int(CCDitem["TEXPMS"])/1000.0)# tot dark current in electrons
    #Then based on how large the dark current is , decide on whether to use 0D or 2D subtraction
    if totdarkcurrent.mean() > CCDunit.dc_2D_limit:
        totdarkcurrent = (
            CCDunit.darkcurrent2D(CCDitem["temperature"], CCDitem["GAIN Mode"])
            * int(CCDitem["TEXPMS"])
            / 1000.0
            )  

    dark_calc_image = (
        CCDunit.ampcorrection
        * totdarkcurrent
        / CCDunit.alpha_avr(CCDitem["GAIN Mode"])
    )


    if (
        (CCDitem["NCSKIP"] > 1)
        or (CCDitem["NCSKIP"] > 1)
        or (CCDitem["NCBIN CCDColumns"] > 1)
        or (CCDitem["NCBIN FPGAColumns"] > 1)
        or (CCDitem["NRBIN"] > 1)
        ):  #
        dark_img = bin_image_with_BC(CCDitem, dark_calc_image)
    else:
        dark_img= dark_calc_image

    return dark_img



def bin_image_with_BC(CCDitem, image_nonbinned=None):
    """
    This is a function to bin an image without any offset or blanks. 

    Args:
        CCDitem:  dictonary containing CCD image and information
        image_nonbinned (optional): numpy array image

    Returns: 
        binned_image: binned image (by summing) 

    """
    if image_nonbinned is None:
        image_nonbinned = CCDitem["IMAGE"]    
    
    totbin=int(CCDitem["NRBIN"])*int(CCDitem["NCBIN CCDColumns"])*int(CCDitem["NCBIN FPGAColumns"])
    
    sumbinned_image=totbin*meanbin_image_with_BC(CCDitem, image_nonbinned)
    
    return sumbinned_image

def meanbin_image_with_BC(CCDitem, image_nonbinned=None):
    """
    This is a function to mean-bin an image without any offset or blanks. Bad columns are skipped.
    This code is used for an image which has already been treated with regards to offset and BC(get_true_image)
    For instance when binning the flatfield image. Code is modified from Georgis predict_image 


    Args:
        CCDitem:  dictonary containing CCD image and information
        image_nonbinned (optional): numpy array image

    Returns: 
        meanbinned_image: binned image (by taking the average) according to the info in CCDitem 

    """

            
    if image_nonbinned is None:
        image_nonbinned = CCDitem["IMAGE"]

    ncol = int(CCDitem["NCOL"]) + 1
    nrow = int(CCDitem["NROW"])

    nrowskip = int(CCDitem["NRSKIP"])
    ncolskip = int(CCDitem["NCSKIP"])

    nrowbin = int(CCDitem["NRBIN"])
    ncolbinC = int(CCDitem["NCBIN CCDColumns"])
    ncolbinF = int(CCDitem["NCBIN FPGAColumns"])

    bad_columns = CCDitem["BC"]

    if nrowbin == 0:  # no binning means beaning of one
        nrowbin = 1

    if ncolbinC == 0:  # no binning means beaning of one
        ncolbinC = 1

    if ncolbinF == 0:  # no binning means beaning of one
        ncolbinF = 1

    ncolbintotal = ncolbinC * ncolbinF
    ntotbin=ncolbintotal*nrowbin


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

                    if (
                        ncolbinC > 1
                        and (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip in bad_columns
                    ):  # +1 becuase Ncol is +1
                        continue
                    else:

                        # add only the actual signal from every pixel 
                        image[j_r, j_c] = (
                            image[j_r, j_c]  
                            + image_nonbinned[
                                (j_r) * nrowbin + j_br + nrowskip,
                                (j_c) * ncolbinC * ncolbinF + j_bc + ncolskip,
                            ]  
                        )

                        nr_of_entries[j_r, j_c] = nr_of_entries[j_r, j_c] + 1

    meanbinned_image = image / nr_of_entries
    return meanbinned_image





    

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






def binning_bc(Ncol, Ncolskip, NcolbinFPGA, NcolbinCCD, BadColumns):
    """
     Routine to estimate the correction factors for column binning with bad columns
     
     
    Args:
        Ncol, Ncolskip, NcolbinFPGA, NcolbinCCD, BadColumns.
        as per ICD. BadColumns - array containing the index of bad columns
                  (the index of first column is 0)

    Returns: 
        n_read: array, containing the number of individually read superpixels
               attributing to the given superpixel
               n_coadd:  - array, containing the number of co-added individual pixels

    """

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




