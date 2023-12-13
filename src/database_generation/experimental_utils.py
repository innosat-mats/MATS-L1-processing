#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:17:27 2022

@author: lindamegner

Functions used in analysis images and calibration development but NOT for operational processing.
"""

import pandas as pd
from mats_l1_processing.L1_calibration_functions import get_true_image, desmear_true_image, binning_bc
import numpy as np

#############################################################################
#   Plotting Functions                                                      #
#############################################################################

def plotCCDitem(CCDitem, fig, axis, title="", clim=999, aspect="auto", altvec=None):
    """Plots a CCDitem in a figure, 
    note that optional argument altvec ONLY specifies the upper and lower limit of the y-axis, 
    not the actual altitudes of the image"""

    image = CCDitem["IMAGE"]
    sp = plot_CCDimage(image, fig, axis, title, clim, aspect, altvec=altvec)
    return sp


def plot_CCDimage(image, fig=None, axis=None, title="", clim=None, aspect="auto", altvec=None, borders=False, nrsig=2):
    """Plots a CCD image in a figure, 
    note that optional argument altvec ONLY specifies the upper and lower limit of the y-axis,
    not the actual altitudes of the image"""
    
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axis = fig.subplots(1)
        print('Creating a new figure since no figure is given, this means ignoring any axes input')
    
    
    if altvec is not None:
        sp = axis.imshow(image, cmap="magma", origin="lower", interpolation="none", extent=[0, image.shape[1], altvec[0], altvec[-1]])
        axis.set_ylabel('Altitude')        
        #x_indices = np.arange(image.shape[1])
        #x_grid, y_grid = np.meshgrid(x_indices, altvec)
    else:
        sp = axis.imshow(image, cmap="magma", origin="lower", interpolation="none")
    
    if clim=='minmax':
        clim=[image.min(), image.max()]

    if clim==None:
        [col, row]=image.shape
        if borders:
            #Take the mean and std of the middle of the image,including borders
            mean=image.mean()
            std=image.std()
        else:
            #Take the mean and std of the middle of the image, not boarders
            mean = image[int(col/2-col*4/10):int(col/2+col*4/10), int(row/2-row*4/10):int(row/2+row*4/10)].mean()
            std = image[int(col/2-col*4/10):int(col/2+col*4/10), int(row/2-row*4/10):int(row/2+row*4/10)].std()
        sp.set_clim([mean - nrsig * std, mean + nrsig * std])
    else:
        sp.set_clim(clim)
    fig.colorbar(sp, ax=axis)
    axis.set_title(title)
    axis.set_aspect(aspect)
    return sp


def plot_CCDimage_hmean(fig, axis, image, title="", clim=999):
    yax = range(0, image.shape[0])
    sp = axis.plot(image.mean(axis=1), yax)
    axis.set_title(title)
    return sp


def plot_simple(filename, clim=999):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    pic = np.float64(Image.open(filename))  # read image
    plt.imshow(pic)
    if clim == 999:
        mean = pic.mean()
        std = pic.std()
        plt.clim([mean - 2 * std, mean + 2 * std])
    else:
        plt.clim(clim)
    plt.colorbar()


def diffplot(image1, image2, title1, title2, clim=999, climdiff=999):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1)
    diffimg = image1 - image2
    if clim == 999:
        plot_CCDimage(image1, fig, ax[0], title=title1)
        plot_CCDimage(image2, fig, ax[1], title=title2)
    else:
        plot_CCDimage(image1, fig, ax[0], title=title1, clim=clim)
        plot_CCDimage(image2, fig, ax[1], title=title2, clim=clim)
    if climdiff == 999:
        plot_CCDimage(diffimg, fig, ax[2], title="pic 1-pic2")
    else:
        plot_CCDimage(diffimg, fig, ax[2], title="pic 1-pic2", clim=climdiff)
    return fig




def plot_full_temperature_info(temperaturedata, relativetimedata):
    import matplotlib.pyplot as plt
    HTR1A = temperaturedata[:, 0]
    HTR1B = temperaturedata[:, 1]
    HTR2A = temperaturedata[:, 2]
    HTR2B = temperaturedata[:, 3]
    HTR8A = temperaturedata[:, 4]
    HTR8B = temperaturedata[:, 5]

    plt.plot(relativetimedata / 60.0, HTR1A,
             label="splitter plate, regulation")
    plt.plot(relativetimedata / 60.0, HTR1B, label="splitter plate, measuring")
    plt.plot(relativetimedata / 60.0, HTR2A, label="limb house, regulation")
    plt.plot(relativetimedata / 60.0, HTR2B, label="limb house, measuring")
    plt.plot(relativetimedata / 60.0, HTR8A, label="UV2 CCDn")
    plt.plot(relativetimedata / 60.0, HTR8B, label="UV1 CCDn")
    plt.xlabel("Time since start of instrument [min]")
    plt.ylabel("Temperature [C]")
    plt.legend()
    plt.show()
    plt.savefig("HTRmeasurements.jpg")
    

#############################################################################
#   Protocol reading functiions                                             # 
#############################################################################

def read_all_files_in_protocol(df, read_from, root_directory):
    from database_generation.read_in_imgview_functions import read_CCDitem_from_imgview
    from mats_l1_processing.read_in_functions import read_CCDitem_image, find_CCDitem_matching_PicID, add_and_rename_CCDitem_info
    from mats_l1_processing.get_temperature import add_rac_temp_data
    import pandas as pd 
    CCDitems = []
    for PicID in list(df["PicID"]):
        if read_from == "rac":
            racdf = pd.read_csv(root_directory + "RacFiles_out/CCD.csv", skiprows=[0]) #Read in full CCD.csv to panda data frame
            CCD_image_data = racdf.to_dict("records") #Comvert full data frame to list of dicts
            CCDitem=find_CCDitem_matching_PicID(CCD_image_data, PicID) #select the dict that corresponds to the wanted PiCID (as given by protocol) and name that dict CCDitem 
            errorflag=read_CCDitem_image(CCDitem, root_directory + "RacFiles_out/") # read in the image that corresponds to the PicID and add that to the CCDitem
            if errorflag:
                raise Exception("Image"+CCDitem['Image File Name'] +"not found") 

                
        elif read_from == "imgview":    
            CCDitem = read_CCDitem_from_imgview(root_directory + "PayloadImages/", PicID)
        else:
            raise Exception("read_from must be rac or imgview")    
        if CCDitem != -999:
            CCDitems.append(CCDitem)
            
    for CCDitem in CCDitems:        
            add_and_rename_CCDitem_info(CCDitem)
            CCDitem["DarkBright"] = df.DarkBright[df.PicID == PicID].iloc[0]
            CCDitem["Shutter"] = df.Shutter[df.PicID == PicID].iloc[0]
            CCDitem["Comment"] = df.Comment[df.PicID == PicID].iloc[0]
            
            #Add temperature data from rac files   
            add_rac_temp_data(root_directory + "RacFiles_out/HTR.csv", CCDitem, labtemp=999)

    return CCDitems

def readprotocol(filename):
    import pandas as pd

    df = pd.read_csv(filename, sep=" ", comment="#",
                     skipinitialspace=True, skiprows=())  
    return df

#############################################################################
#   Forward model  routines                                                 # 
#############################################################################

def desmear_true_image_reverse(header, image=None):
    # add readout offset (pixel blank value) and bad colums stuff,
    # by reversing get_true_image
    from mats_l1_processing.L1_calibration_functions import calculate_time_per_row

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

def get_true_image_reverse(header, true_image=None):
    # add readout offset (pixel blank value) and bad colums stuff,
    # by reversing get_true_image
    from mats_l1_processing.L1_calibration_functions import binning_bc
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

#############################################################################
#   Miscellaneous                                                           # 
#############################################################################


def filter_on_time(CCDitems, starttime=None, stoptime=None):
    I = []
    for i in range(len(CCDitems)):
        image_time = pd.to_datetime(
            CCDitems[i]["EXP Date"], format="%Y-%m-%dT%H:%M:%S.%fZ"
        )
        if (starttime != None) and (stoptime != None):
            if (image_time > starttime) and (image_time < stoptime):
                I.append(i)
        elif (starttime != None) and (stoptime == None):
            if image_time > starttime:
                I.append(i)
        elif (starttime == None) and (stoptime != None):
            if image_time < starttime:
                I.append(i)
        else:
            Warning("Start or end time invalid")

    CCDitems = [CCDitems[i] for i in I]
    return CCDitems


#############################################################################
#   Georgis code                                                           # 
#############################################################################

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


def calibrate_CCDitems(CCDitems,instrument, plot=False):
    """
    Calibrate all CCDitems in the list

    Parameters
    ----------
    CCDitems : List of dictionaries
        Contains images and housing data
    instrument: instrument object, see mats_l1_processing.instrument
    plot : logical, optional
        If true the calibrations steps are plotted. The default is False.

    Returns
    -------
    Does not return anything but CCDitems will now contain calibrated images
    """
    from mats_l1_processing.L1_calibrate import L1_calibrate
    import matplotlib.pyplot as plt
    
    
    for CCDitem in CCDitems:
        (
            image_lsb,
            image_bias_sub,
            image_desmeared,
            image_dark_sub,
            image_calib_nonflipped,
            image_calibrated,
            errors
        ) = L1_calibrate(CCDitem, instrument)

        if plot:
            fig, ax = plt.subplots(5, 1)
            plot_CCDimage(image_lsb, fig, ax[0], "Original LSB")
            plot_CCDimage(image_bias_sub, fig, ax[1], "Bias subtracted")
            plot_CCDimage(image_desmeared, fig, ax[2], " Desmeared LSB")
            plot_CCDimage(
                image_dark_sub, fig, ax[3], " Dark current subtracted LSB"
            )
            plot_CCDimage(
                image_calib_nonflipped, fig, ax[4], " Flat field compensated LSB"
            )
            fig.suptitle(CCDitem["channel"])
    