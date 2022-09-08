#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 08:57:37 2020

@author: lindamegner

Functions used to read in MATS images and data from rac files. 
The other ways to read in (From KTH, from Immage viewer) is being moved to read_imgview_functions.py in database_generation. 
The housekeepting data temperatures can also be read in using these functions.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import json
from PIL import Image


# import imagereader


def plot_full_temperature_info(temperaturedata, relativetimedata):
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


def add_temperature_info_to_CCDitems(CCDitems, read_from, directory, labtemp=999):
    from mats_l1_processing.get_temperature import create_temperature_info_array, add_temperature_info

    if read_from == "rac":
        temperaturedata, relativetimedata = create_temperature_info_array(
            directory + "RacFiles_out/HTR.csv"
        )
    elif read_from != "rac":
        temperaturedata = 999
        relativetimedata = 999

    # plot_full_temperature_info(temperaturedata,relativetimedata)

    for CCDitem in CCDitems:
        CCDitem = add_temperature_info(
            CCDitem, temperaturedata, relativetimedata, labtemp
        )

    return CCDitems


def read_all_files_in_root_directory(read_from, root_directory):
    # Reads in file from differnt sub directories depending on what read_from is set to.
    if read_from == "rac":
        CCDitems = read_all_files_in_directory(
            read_from, root_directory+'RacFiles_out/')
    elif read_from == "rac_operational":
        CCDitems = read_all_files_in_directory(
            read_from, root_directory)
    elif read_from == "imgview":
        CCDitems = read_all_files_in_directory(
            read_from, root_directory+'PayloadImages/')
    else:
        raise Exception("read_from needs to = rac,rac_operational or imgview ")
    return CCDitems


def read_all_files_in_directory(read_from, directory):
    from .get_temperature import add_rac_temp_data
    from database_generation.read_in_imgview_functions import read_CCDitem_from_imgview
    import os
    import pandas as pd
    CCDitems = []
    if (read_from == "rac") or (read_from == "rac_operational"):
        df = pd.read_csv(directory + "CCD.csv", skiprows=[0])
        items = df.to_dict("records")
        for item in items:
            errorflag = read_CCDitem_image(item, directory)
            if errorflag:
                print("Warning, the image file: ",
                      item['Image File Name'], " cannot be found or read and has been removed")
            else:
                CCDitems.append(item)
    elif read_from == "imgview":
        for file in os.listdir(directory):
            if len(file) > 11 and file.endswith("_output.txt"):
                IDstring = file[:-11]
                CCDitem = read_CCDitem_from_imgview(directory, IDstring)
                CCDitems.append(CCDitem)
    else:
        raise Exception("read_from needs to = rac,rac_operational or imgview ")

    for CCDitem in CCDitems:

        add_and_rename_CCDitem_info(CCDitem)

        if read_from == "rac":  # Add temperature data from rac files
            add_rac_temp_data(directory + "/HTR.csv", CCDitem, labtemp=999)

    return CCDitems


def add_and_rename_CCDitem_info(CCDitem):
    import numpy as np
    from math import log
    from math import isnan
#    from .get_temperature import create_temperature_info_array, add_temperature_info

    if CCDitem["read_from"] == "rac":
        # LM 201113 In old versions, as such of TVAC in summer 2019 this was read in as float not integer, hence convert

        if (
            type(CCDitem["EXP Nanoseconds"]) is float
            and isnan(CCDitem["EXP Nanoseconds"]) == False
        ):
            CCDitem["EXP Nanoseconds"] = int(CCDitem["EXP Nanoseconds"])

        # LM 201113 In old versions, as such of TVAC in summer 2019 this was read in as float not integer, hence convert
        if (type(CCDitem["CCDSEL"]) is float) and (
            isnan(CCDitem["CCDSEL"]) == False
        ):  # reads data from all images (itemnumbers) in the rac file

            #    CCD_image_data=read_MATS_image(rac_dir+'images.json') #lest of dictionries
            CCDitem["CCDSEL"] = int(CCDitem["CCDSEL"])

    try:
        CCDitem["EXP Nanoseconds"]
    except:
        try:
            CCDitem["EXP Nanoseconds"] = 1.0e9*(
                int(CCDitem["EXPTS"]) + int(CCDitem["EXPTSS"]) / 2 ** 16)
        except:
            raise Exception("No info on the relative time")

    if CCDitem["read_from"] == "rac":
        #   Renaming of stuff. The names in the code here is based on the old rac extract file (prior to May 2020) rac_extract file works
        CCDitem["id"] = str(CCDitem["EXP Nanoseconds"]) + \
            "_" + str(CCDitem["CCDSEL"])  #CCDitem["id"] should not be needed in operational retrieval. Keeping it because protocol reading / CodeCalibrationReport needs it.  LM220908

    try:
        CCDitem["NCBIN CCDColumns"]
    except:
        CCDitem["NCBIN CCDColumns"] = CCDitem["NColBinCCD"]


    try:
        CCDitem["NCBIN FPGAColumns"]
    except:
        CCDitem["NCBIN FPGAColumns"] = 2**CCDitem["NColBinFPGA"]


    if CCDitem["read_from"] == "rac":

        try:
            CCDitem["DigGain"]
        except:
            CCDitem["DigGain"] = CCDitem["GAIN Truncation"]
        try:
            CCDitem["SigMode"]
            # 0 should be high in output LM 200604. This is opposite to what it is in input.
            # TODO change all L1 code with so that it uses CCDitem["GAIN Mode"] instead of SigMode LM 211022
        except:
            if CCDitem["GAIN Mode"] == 'High':
                CCDitem["SigMode"] = 0
            elif CCDitem["GAIN Mode"] == 'Low':
                CCDitem["SigMode"] = 1

    # TODO rename. See teble 2 in Paylaod internal ICD.Row 11 in table.  High gain mode= signal mode 0. Low gain mode= sig mode 1.
    # Gain timing= full/fast timing
    # Gain truncation = gain (bit truncated in FPGA)=Dig gain

        # Convert BC to list of integer instead of str
        if CCDitem["BC"] == "[]":
            CCDitem["BC"] = np.array([])
        else:
            strlist = CCDitem["BC"][1:-1].split(" ")
            CCDitem["BC"] = np.array([int(i) for i in strlist])

    if CCDitem["read_from"] == "imgview":
    #  Hack to have no compensation for bad colums at the time. TODO later.
        if CCDitem['NBC']==0:
            CCDitem['BC']=np.array([])  
    
        
    CCDitem["channel"] = channel_num_to_str(CCDitem["CCDSEL"])

    # Add temperature info fom OBC, the temperature info from the rac files are better since they are based on the thermistos on hte UV channels
    ADC_temp_in_mV = int(CCDitem["TEMP"]) / 32768 * 2048
    ADC_temp_in_degreeC = 1.0 / 0.85 * ADC_temp_in_mV - 296
    CCDitem["OBCtemperature"] = ADC_temp_in_degreeC

    return CCDitem


def readprotocol(filename):
    import pandas as pd

    df = pd.read_csv(filename, sep=" ", comment="#",
                     skipinitialspace=True, skiprows=())
    return df


def searchlist(list, key, value):
    found = False
    for item in list:
        if item[key] == value:
            found = True
            return item

    if not found:
        print("Warning: Item not found")


def read_CCDitem_image(item, rac_dir, extract_images=True, labtemp=999):
    from PIL import Image

    errorflag = False
    pngfile = rac_dir + str(item["Image File Name"])
    jsonfile = pngfile[0:-4] + ".json"
    try:
        if extract_images:
            item["IMAGE"] = np.float64(Image.open(pngfile))
        else:
            item["IMAGE"] = []
        with open(jsonfile) as f:
            item["jsondata"] = json.load(f)

        item["read_from"] = "rac"

    except:
        print("Warning, the image file: ", pngfile, " cannot be found or read")
        errorflag = True

    return errorflag


def find_CCDitem_matching_protocol_entry(CCD_image_data, PicID):
    # reads data from one image (itemnumber) in the rac file

    if PicID.count("_") == 1:  # new way of naming as of June 2020 in protocol
        itemnumber = int(PicID[:-2])
        CCDSEL = int(PicID[-1:])
        try:
            CCDitem = next(
                item
                for item in CCD_image_data
                # LM 211025 Edited so that the last three digits are ignored
                # (5 digits corresponds to 10^5 ns) since computations
                # of the number of nanoseconds are rounded off in teh new imageviewer
                # (which must have been updated in ) summer 2021.
                if round(item["EXP Nanoseconds"],-5) == round(itemnumber,-5) and item["CCDSEL"] == CCDSEL
            )
        except:
            raise Exception('Image '+PicID +
                            ' in protocol but not found in rac file')
    elif PicID.count("_") == 2:  # old way of naming as of spring 2020 in protociol
        itemnumber = int(PicID.split("_")[0])
        CCDSEL = int(PicID[-1:])
        try:
            CCDitem = next(
                item
                for item in CCD_image_data
                if str(item["EXP Nanoseconds"])[:-9] == str(itemnumber)
                and item["CCDSEL"] == CCDSEL
            )
        except:
            raise Exception(
                'Image'+PicID + ' in protocol but not found in rac file')
    else:
        raise Exception("strange naming in protocol, PicID=", PicID)

    return CCDitem


def read_CCDitem_rac_or_imgview(dirname, PicID, read_from):
    from database_generation.read_in_imgview_functions import read_CCDitem_from_imgview
    if read_from == 'rac':
        CCDitem = read_CCDitem(dirname+'RacFiles_out/', PicID)
    elif read_from == 'imgview':
        CCDitem = read_CCDitem_from_imgview(dirname+'PayloadImages/', PicID)
    else:
        raise Exception('read_from is not defined')
    return CCDitem


def read_CCDitem(rac_dir, PicID, labtemp=999):
    # reads data from one image (itemnumber) in the rac file
    import pandas as pd
#    from mats_l1_processing.read_in_functions import read_CCDitem_image, find_CCDitem_matching_protocol_entry, add_and_rename_CCDitem_info
    from .get_temperature import add_rac_temp_data

    df = pd.read_csv(rac_dir + "CCD.csv", skiprows=[0])
    CCD_image_data = df.to_dict("records")

    CCDitem = find_CCDitem_matching_protocol_entry(CCD_image_data, PicID)
    errorflag = read_CCDitem_image(CCDitem, rac_dir)
    if errorflag:
        raise Exception("Image"+CCDitem['Image File Name'] + "not found")

    add_and_rename_CCDitem_info(CCDitem)

    # Add temperature data from rac files
    add_rac_temp_data(rac_dir + "/HTR.csv", CCDitem, labtemp=999)

    return CCDitem


def read_CCDitems(directory):
    read_from = 'rac'
    CCDitems = read_all_files_in_directory(read_from, directory)
    return CCDitems


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [
        bind.get(itm, None) for itm in a
    ]  # None can be replaced by any other "not in b" value


def channel_num_to_str(ccdsel):
    # Assign string names to all channels
    if ccdsel == 1:
        channel = "IR1"
    elif ccdsel == 4:
        channel = "IR2"
    elif ccdsel == 3:
        channel = "IR3"
    elif ccdsel == 2:
        channel = "IR4"
    elif ccdsel == 5:
        channel = "UV1"
    elif ccdsel == 6:
        channel = "UV2"
    elif ccdsel == 7:
        channel = "NADIR"
    else:
        print("Error in CCDSEL, CCDSEL=", ccdsel)

    return channel
