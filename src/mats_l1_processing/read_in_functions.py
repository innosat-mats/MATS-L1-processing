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

    plt.plot(relativetimedata / 60.0, HTR1A, label="splitter plate, regulation")
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
    from get_temperature import create_temperature_info_array, add_temperature_info

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
    #        timestamp=epoch+datetime.timedelta(0,CCDitem['reltime'])

    return CCDitems


def read_all_files_in_directory(read_from, directory):
    if read_from == "rac":
        CCDitems = read_CCDitems(directory)
    else:
        raise Exception("read_from = imgview is not yet implemented")
    return CCDitems


def readprotocol(filename):
    import pandas as pd

    df = pd.read_csv(filename, sep=" ", comment="#", skipinitialspace=True, skiprows=())
    return df


def searchlist(list, key, value):
    found = False
    for item in list:
        if item[key] == value:
            found = True
            return item

    if not found:
        print("Warning: Item not found")


def read_MATS_image(rac_dir, extract_images=True):
    import pandas as pd
    from PIL import Image
    import math

    # If you use an newer version of the rac extract reader then pathdir is not needed. Nwwer is approximately from the start of 2020.
    #    json_file = open(filename,'r')
    #    CCD_image_data = json.load(json_file)
    #    json_file.close

    df = pd.read_csv(rac_dir + "CCD.csv", skiprows=[0])
    CCD_image_data = df.to_dict("records")

    for item in CCD_image_data:

        # LM 201113 In old versions, as such of TVAC in summer 2019 this was read in as float not integer, hence convert

        if (
            type(item["EXP Nanoseconds"]) is float
            and math.isnan(item["EXP Nanoseconds"]) == False
        ):
            item["EXP Nanoseconds"] = int(item["EXP Nanoseconds"])

        # LM 201113 In old versions, as such of TVAC in summer 2019 this was read in as float not integer, hence convert
        if (type(item["CCDSEL"]) is float) and (
            math.isnan(item["CCDSEL"]) == False
        ):  # reads data from all images (itemnumbers) in the rac file

            #    CCD_image_data=read_MATS_image(rac_dir+'images.json') #lest of dictionries
            item["CCDSEL"] = int(item["CCDSEL"])

        pngfile = rac_dir + str(item["Image File Name"])

        jsonfile = pngfile[0:-4] + ".json"
        try:
            if extract_images:
                item["IMAGE"] = np.float64(Image.open(pngfile))
            else:
                item["IMAGE"] = []
            with open(jsonfile) as f:
                item["jsondata"] = json.load(f)
        except:
            print("Warning, the image file: ", pngfile, " cannot be found or read")
            CCD_image_data.remove(item)

    return CCD_image_data


def read_CCDitem(rac_dir, PicID, labtemp=999):
    # reads data from one image (itemnumber) in the rac file
    from math import log
    import pandas as pd
    from PIL import Image
    from .get_temperature import create_temperature_info_array, add_temperature_info

    df = pd.read_csv(rac_dir + "CCD.csv", skiprows=[0])
    CCD_image_data = df.to_dict("records")

    if PicID.count("_") == 1:  # new way of naming as of June 2020 in protocol
        itemnumber = int(PicID[:-2])
        CCDSEL = int(PicID[-1:])
        CCDitem = next(
            item
            for item in CCD_image_data
            if item["EXP Nanoseconds"] == itemnumber and item["CCDSEL"] == CCDSEL
        )
    elif PicID.count("_") == 2:  # old way of naming as of spring 2020 in protociol
        itemnumber = int(PicID.split("_")[0])
        CCDSEL = int(PicID[-1:])

        CCDitem = next(
            item
            for item in CCD_image_data
            if str(item["EXP Nanoseconds"])[:-9] == str(itemnumber)
            and item["CCDSEL"] == CCDSEL
        )
    else:
        raise Exception("strange naming in protocol, PicID=", PicID)

    if int(CCDitem["CCDSEL"]) == 1:  # input CCDSEL=1
        channel = "IR1"
    elif int(CCDitem["CCDSEL"]) == 4:  # input CCDSEL=8
        channel = "IR2"
    elif int(CCDitem["CCDSEL"]) == 3:  # input CCDSEL=4
        channel = "IR3"
    elif int(CCDitem["CCDSEL"]) == 2:  # input CCDSEL=2
        channel = "IR4"
    elif int(CCDitem["CCDSEL"]) == 5:  # input CCDSEL=16
        channel = "UV1"
    elif int(CCDitem["CCDSEL"]) == 6:  # input CCDSEL=32
        channel = "UV2"
    elif int(CCDitem["CCDSEL"]) == 7:  # input CCDSEL=64
        channel = "NADIR"
    else:
        print("Error in CCDSEL, CCDSEL=", int(CCDitem["CCDSEL"]))

    CCDitem["channel"] = channel
    #   Renaming of stuff. The names in the code here is based on the old rac extract file (prior to May 2020) rac_extract file works
    CCDitem["id"] = str(CCDitem["EXP Nanoseconds"]) + "_" + str(CCDitem["CCDSEL"])

    # TODO LM June 2020: Change  all code so that the new names, i. CCDitem['NCBIN CCDColumns'] and CCDitem['NCBIN FPGAColumns'] are used instead of the old.
    try:
        CCDitem["NColBinCCD"]
    except:
        CCDitem["NColBinCCD"] = CCDitem["NCBIN CCDColumns"]

    # CCDitem['NColBinFPGA']=CCDitem['NCBIN FPGAColumns']
    try:
        CCDitem["NColBinFPGA"]
    except:
        CCDitem["NColBinFPGA"] = log(CCDitem["NCBIN FPGAColumns"]) / log(2)

        # del CCDitem['NCBIN FPGAColumns']
    if CCDitem["GAIN Mode"] == "High":
        CCDitem["DigGain"] = 0
    elif CCDitem["GAIN Mode"] == "Low":
        CCDitem["DigGain"] = 1
    else:
        raise Exception("GAIN mode set to strange value")

    CCDitem["SigMode"] = 0
    # This should be read in, 0 should be high in output LM 200604
    #       CCDitem['']=CCDitem['']

    CCDitem["read_from"] = "rac"
    try:
        CCDitem["reltime"] = 1.0e-9 * CCDitem["EXP Nanoseconds"]
    except:
        try:
            CCDitem["reltime"] = (
                int(CCDitem["EXPTS"]) + int(CCDitem["EXPTSS"]) / 2 ** 16
            )
        except:
            raise Exception("No info on the relative time")

    #        print(pathdir+str(CCD_image_data[i]['IMAGEFILE']) + '_data.npy')
    pngfile = rac_dir + str(CCDitem["Image File Name"])
    jsonfile = pngfile[0:-4] + ".json"
    try:
        CCDitem["IMAGE"] = np.float64(Image.open(pngfile))
        with open(jsonfile) as f:
            CCDitem["jsondata"] = json.load(f)

    except:
        print("Warning, the image file: ", pngfile, " cannot be found or read")

        # Added temperature read in

    if CCDitem["read_from"] == "rac":
        temperaturedata, relativetimedata = create_temperature_info_array(
            rac_dir + "HTR.csv"
        )
    else:
        temperaturedata = 999
        relativetimedata = 999
        raise Exception(
            "Procedure CCDitem should never be called when not read_from not equals rac "
        )

    # plot_full_temperature_info(temperaturedata,relativetimedata)

    CCDitem = add_temperature_info(CCDitem, temperaturedata, relativetimedata, labtemp)

    return CCDitem


def read_CCDitemsx(rac_dir, pathdir):
    # reads data from all images (itemnumbers) in the rac file

    CCD_image_data = read_MATS_image(rac_dir + "images.json", pathdir)

    for CCDitem in CCD_image_data:
        #        CCDitem=CCD_image_data[itemnumber]

        # The variables below are remain question marks
        #    CCDitem['Noverflow'] = Noverflow # FBINOV?
        #   CCDitem['Ending'] = Ending       #not found in rac

        #    CCDitem['BC']=CCDitem['BC'][0] Bad culums ska vara en array
        CCDitem["CCDSEL"] = CCDitem["CCDSEL"][0]
        CCDitem["EXPTS"] = CCDitem["EXPTS"][0]
        CCDitem["EXPTSS"] = CCDitem["EXPTSS"][0]
        CCDitem["FBINOV"] = CCDitem["FBINOV"][0]
        CCDitem["FRAME"] = CCDitem["FRAME"][0]
        CCDitem["JPEGQ"] = CCDitem["JPEGQ"][0]
        CCDitem["LBLNK"] = CCDitem["LBLNK"][0]
        CCDitem["NBC"] = CCDitem["NBC"][0]
        CCDitem["NCOL"] = CCDitem["NCOL"][0]
        CCDitem["NCSKIP"] = CCDitem["NCSKIP"][0]
        CCDitem["NFLUSH"] = CCDitem["NFLUSH"][0]
        CCDitem["NRBIN"] = CCDitem["NRBIN"][0]
        CCDitem["NROW"] = CCDitem["NROW"][0]
        CCDitem["NRSKIP"] = CCDitem["NRSKIP"][0]
        CCDitem["SID_mnemonic"] = CCDitem["SID_mnemonic"][0]
        CCDitem["TBLNK"] = CCDitem["TBLNK"][0]
        CCDitem["TEMP"] = CCDitem["TEMP"][0]
        CCDitem["TEXPMS"] = CCDitem["TEXPMS"][0]
        CCDitem["TIMING1"] = CCDitem["TIMING1"][
            0
        ]  # Named Reserved1 in Georgis code /LM 20191115
        CCDitem["TIMING2"] = CCDitem["TIMING2"][
            0
        ]  # Named Reserved2 in Georgis code /LM 20191115
        CCDitem["TIMING3"] = CCDitem["TIMING3"][
            0
        ]  # Named VersionDate in Georgis code /LM 20191115
        CCDitem["VERSION"] = CCDitem["VERSION"][0]
        CCDitem["WDWOV"] = CCDitem["WDWOV"][0]
        CCDitem["ZERO"] = CCDitem["ZERO"][0]

        CCDitem["NColBinFPGA"] = CCDitem["NColBinFPGA"][0]
        CCDitem["NColBinCCD"] = CCDitem["NColBinCCD"][0]
        CCDitem["DigGain"] = CCDitem["DigGain"][0]
        CCDitem["TimingFlag"] = CCDitem["TimingFlag"][0]
        CCDitem["SigMode"] = CCDitem["SigMode"][0]
        CCDitem["WinModeFlag"] = CCDitem["WinModeFlag"][0]
        CCDitem["WinMode"] = CCDitem["WinMode"][0]

        if int(CCDitem["CCDSEL"]) == 1:
            channel = "IR1"
        elif int(CCDitem["CCDSEL"]) == 4:
            channel = "IR2"
        elif int(CCDitem["CCDSEL"]) == 3:
            channel = "IR3"
        elif int(CCDitem["CCDSEL"]) == 2:
            channel = "IR4"
        elif int(CCDitem["CCDSEL"]) == 5:
            channel = "UV1"
        elif int(CCDitem["CCDSEL"]) == 6:
            channel = "UV2"
        elif int(CCDitem["CCDSEL"]) == 7:
            channel = "NADIR"
        else:
            print("Error in CCDSEL, CCDSEL=", int(CCDitem["CCDSEL"]))

        CCDitem["channel"] = channel

    return CCD_image_data


def read_CCDitems(rac_dir, labtemp=999):
    from math import log
    from math import isnan
    from .get_temperature import create_temperature_info_array, add_temperature_info

    CCD_image_data = read_MATS_image(rac_dir)

    # Throw out items that have not been properly read:
    for CCDitem in CCD_image_data:
        if isnan(CCDitem["CCDSEL"]):
            CCD_image_data.remove(CCDitem)

    for CCDitem in CCD_image_data:

        # CCDitem=CCD_image_data[itemnumber]

        # The variables below are remain question marks
        #    CCDitem['Noverflow'] = Noverflow # FBINOV?
        #   CCDitem['Ending'] = Ending       #not found in rac

        #        CCDitem['TIMING1']=CCDitem['TIMING1'][0] # Named Reserved1 in Georgis code /LM 20191115
        #        CCDitem['TIMING2']=CCDitem['TIMING2'][0] # Named Reserved2 in Georgis code /LM 20191115
        #        CCDitem['TIMING3']=CCDitem['TIMING3'][0] # Named VersionDate in Georgis code /LM 20191115

        # # The parameters below are missing in the new version of the rac files /LM 200603
        # CCDitem['SID_mnemonic']=CCDitem['SID_mnemonic'][0]
        # CCDitem['DigGain']=CCDitem['DigGain'][0]
        # CCDitem['TimingFlag']=CCDitem['TimingFlag'][0]
        # CCDitem['SigMode']=CCDitem['SigMode'][0]
        # CCDitem['WinModeFlag']=CCDitem['WinModeFlag'][0]
        # CCDitem['WinMode']=CCDitem['WinMode'][0]

        if int(CCDitem["CCDSEL"]) == 1:  # input CCDSEL=1
            channel = "IR1"
        elif int(CCDitem["CCDSEL"]) == 4:  # input CCDSEL=8
            channel = "IR2"
        elif int(CCDitem["CCDSEL"]) == 3:  # input CCDSEL=4
            channel = "IR3"
        elif int(CCDitem["CCDSEL"]) == 2:  # input CCDSEL=2
            channel = "IR4"
        elif int(CCDitem["CCDSEL"]) == 5:  # input CCDSEL=16
            channel = "UV1"
        elif int(CCDitem["CCDSEL"]) == 6:  # input CCDSEL=32
            channel = "UV2"
        elif int(CCDitem["CCDSEL"]) == 7:  # input CCDSEL=64
            channel = "NADIR"
        else:
            print("Error in CCDSEL, CCDSEL=", int(CCDitem["CCDSEL"]))

        CCDitem["channel"] = channel
        #       Renaming of stuff. The names in the code here is based on the old rac extract file (prior to May 2020) rac_extract file works
        CCDitem["id"] = str(CCDitem["EXP Nanoseconds"]) + "_" + str(CCDitem["CCDSEL"])

        # TODO LM June 2020: Change  all code so that the new names, i. CCDitem['NCBIN CCDColumns'] and CCDitem['NCBIN FPGAColumns'] are used instead of the old.
        try:
            CCDitem["NColBinCCD"]
        except:
            CCDitem["NColBinCCD"] = CCDitem["NCBIN CCDColumns"]

        # CCDitem['NColBinFPGA']=CCDitem['NCBIN FPGAColumns']
        try:
            CCDitem["NColBinFPGA"]
        except:
            CCDitem["NColBinFPGA"] = log(CCDitem["NCBIN FPGAColumns"]) / log(2)

        # del CCDitem['NCBIN FPGAColumns']
        if CCDitem["GAIN Mode"] == "High":
            CCDitem["DigGain"] = 0
        elif CCDitem["GAIN Mode"] == "Low":
            CCDitem["DigGain"] = 1
        else:
            raise Exception("GAIN mode set to strange value")

        CCDitem["SigMode"] = 0
        # This should be read in, 0 should be high in output LM 200604
        #       CCDitem['']=CCDitem['']
        CCDitem["read_from"] = "rac"
        try:
            CCDitem["reltime"] = 1.0e-9 * CCDitem["EXP Nanoseconds"]
        except:
            try:
                CCDitem["reltime"] = (
                    int(CCDitem["EXPTS"]) + int(CCDitem["EXPTSS"]) / 2 ** 16
                )
            except:
                raise Exception("No info on the relative time")

        # Convert BC to list of integer instead of str
        if CCDitem["BC"] == "[]":
            CCDitem["BC"] = np.array([])
        else:
            strlist = CCDitem["BC"][1:-1].split(" ")
            CCDitem["BC"] = np.array([int(i) for i in strlist])

        # Added temperature read in

    if CCDitem["read_from"] == "rac":
        temperaturedata, relativetimedata = create_temperature_info_array(
            rac_dir + "HTR.csv"
        )
    elif CCDitem["read_from"] != "rac":
        temperaturedata = 999
        relativetimedata = 999

    # plot_full_temperature_info(temperaturedata,relativetimedata)

    for CCDitem in CCD_image_data:

        CCDitem = add_temperature_info(
            CCDitem, temperaturedata, relativetimedata, labtemp
        )
    #        timestamp=epoch+datetime.timedelta(0,CCDitem['reltime'])

    return CCD_image_data


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [
        bind.get(itm, None) for itm in a
    ]  # None can be replaced by any other "not in b" value
