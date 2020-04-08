#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:09:57 2019

@author: franzk, Linda Megner
(Linda has removed some fuctions and added more . Original franzk script is L1_functions.py)


functions used for MATS L1 processing, based on corresponding MATLAB scripts provided by
Georgi Olentsenko and Mykola Ivchenko
The MATLAB script can be found here: https://github.com/OleMartinChristensen/MATS-image-analysis



"""

import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.io
import json
from PIL import Image

# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/lindamegner/MATS/retrieval/Level0/MATS-L0-processing-master')

import imagereader as imagereader
#ismember function taken from https://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function
#needs to be tested, if actually replicates matlab ismember function

def readimageviewpic(dirname,picnr,rawflag):
# Reads data from output image file and combines it with the txt file.
# Note that this is the old version that reads the onld style of filenames. 
# The new version of this function is read_pnm_image_and_txt      
    if rawflag==1:
        imagefile= dirname +'rawoutput'+str(picnr) +'.pnm'
    else:
        imagefile= dirname +'output'+str(picnr) +'.pnm'
    txtfile=dirname +'output'+str(picnr) +'.txt'
    image_raw = np.int64(Image.open(imagefile))
    CCDitem=read_txtfile_create_CCDitem(txtfile)
    CCDitem['IMAGE']=image_raw    
    return CCDitem


def readimageviewpic2(dirname,IDstring,rawflag=0):
# Reads data from output image file and combines it with the txt file.
# Note that this is the old version that reads the onld style of filenames. 
# The new version of this function is read_pnm_image_and_txt      
    imagefile= dirname + '/'+ IDstring +'.pnm'
    txtfile= dirname +'/'+ IDstring +'_output.txt'
    image_raw = np.int64(Image.open(imagefile))
    CCDitem=read_txtfile_create_CCDitem(txtfile)
    CCDitem['IMAGE']=image_raw    
    return CCDitem

def readimageviewpics(dirname,rawflag, filelist=[]):
    from os import listdir
    from os.path import isfile, join    
    #Reads  all images in directory
    all_files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
    pnm_files = list(filter(lambda x: x[-4:] == '.pnm', all_files))
    CCDitems=[]
    for pnm_file in pnm_files:
        IDstring=pnm_file[:-4]
        CCDitem=readimageviewpic2(dirname,IDstring,rawflag)
        CCDitems.append(CCDitem)

    return CCDitems


def readselectedimageviewpics(dirname,IDlist):
    CCDitems=[]
    for IDstring in IDlist:
        CCDitem=readimageviewpic2(dirname,IDstring)
        CCDitems.append(CCDitem)
    return CCDitems




def read_pnm_image_and_txt(dirname,picid):
# Reads data from output image file and combines it with the txt file.
    imagefile= dirname +picid +'.pnm'
    txtfile=dirname +picid+'_output.txt'
    image_raw = np.int64(Image.open(imagefile))
    CCDitem=read_txtfile_create_CCDitem(txtfile)
    CCDitem['IMAGE']=image_raw    
    return CCDitem

def read_MATS_image(filename,pathdir=''):
    json_file = open(filename,'r')
    CCD_image_data = json.load(json_file)
    json_file.close
            
    for i in range(len(CCD_image_data)):
        CCD_image_data[i]['IMAGE'] = np.load(pathdir+str(CCD_image_data[i]['IMAGEFILE']) + '_data.npy')

    return CCD_image_data


def read_CCDitem(rac_image_json_file,rac_sub_dir,itemnumber,rac_image_json_dir=''):
# reads data from one image (itemnumber) in the rac file

    CCD_image_data=read_MATS_image(rac_image_json_dir+rac_sub_dir+rac_image_json_file,rac_image_json_dir)

    CCDitem=CCD_image_data[itemnumber]
 

# The variables below are remain question marks
#    CCDitem['Noverflow'] = Noverflow # FBINOV?
 #   CCDitem['Ending'] = Ending       #not found in rac 
    


    
#    CCDitem['BC']=CCDitem['BC'][0] Bad culums ska vara en array
    CCDitem['CCDSEL']=CCDitem['CCDSEL'][0]
    CCDitem['EXPTS']=CCDitem['EXPTS'][0]
    CCDitem['EXPTSS']=CCDitem['EXPTSS'][0]
    CCDitem['FBINOV']=CCDitem['FBINOV'][0]
    CCDitem['FRAME']=CCDitem['FRAME'][0]
    CCDitem['JPEGQ']=CCDitem['JPEGQ'][0]
    CCDitem['LBLNK']=CCDitem['LBLNK'][0]
    CCDitem['NBC']=CCDitem['NBC'][0]
    CCDitem['NCOL']=CCDitem['NCOL'][0]
    CCDitem['NCSKIP']=CCDitem['NCSKIP'][0]
    CCDitem['NFLUSH']=CCDitem['NFLUSH'][0]
    CCDitem['NRBIN']=CCDitem['NRBIN'][0]        
    CCDitem['NROW']=CCDitem['NROW'][0]
    CCDitem['NRSKIP']=CCDitem['NRSKIP'][0]
    CCDitem['SID_mnemonic']=CCDitem['SID_mnemonic'][0]
    CCDitem['TBLNK']=CCDitem['TBLNK'][0]
    CCDitem['TEMP']=CCDitem['TEMP'][0]
    CCDitem['TEXPMS']=CCDitem['TEXPMS'][0]
    CCDitem['TIMING1']=CCDitem['TIMING1'][0] # Named Reserved1 in Georgis code /LM 20191115
    CCDitem['TIMING2']=CCDitem['TIMING2'][0] # Named Reserved2 in Georgis code /LM 20191115
    CCDitem['TIMING3']=CCDitem['TIMING3'][0] # Named VersionDate in Georgis code /LM 20191115
    CCDitem['VERSION']=CCDitem['VERSION'][0]
    CCDitem['WDWOV']=CCDitem['WDWOV'][0]
    CCDitem['ZERO']=CCDitem['ZERO'][0] 

    CCDitem['NColBinFPGA']=CCDitem['NColBinFPGA'][0]
    CCDitem['NColBinCCD']=CCDitem['NColBinCCD'][0]    
    CCDitem['DigGain']=CCDitem['DigGain'][0]
    CCDitem['TimingFlag']=CCDitem['TimingFlag'][0] 
    CCDitem['SigMode']=CCDitem['SigMode'][0]
    CCDitem['WinModeFlag']=CCDitem['WinModeFlag'][0]     
    CCDitem['WinMode']=CCDitem['WinMode'][0]  
    


    
    if int(CCDitem['CCDSEL'])==1:
        channel='IR1'
    elif int(CCDitem['CCDSEL'])==4:
        channel='IR2'
    elif int(CCDitem['CCDSEL'])==3:
        channel='IR3'
    elif int(CCDitem['CCDSEL'])==2:
        channel='IR4'
    elif int(CCDitem['CCDSEL'])==5:
        channel='UV1'
    elif int(CCDitem['CCDSEL'])==6:
        channel='UV2'
    elif int(CCDitem['CCDSEL'])==7:
        channel='NADIR'
    else:
        print('Error in CCDSEL, CCDSEL=',int(CCDitem['CCDSEL']))  
        
    
    CCDitem['channel']=channel
    


    return CCDitem


def read_CCDitems(rac_image_json_file,rac_sub_dir,rac_image_json_dir=''):
# reads data from one image (itemnumber) in the rac file

    CCD_image_data=read_MATS_image(rac_image_json_dir+rac_sub_dir+rac_image_json_file,rac_image_json_dir)


    for CCDitem in CCD_image_data:
#        CCDitem=CCD_image_data[itemnumber]
     
    
    # The variables below are remain question marks
    #    CCDitem['Noverflow'] = Noverflow # FBINOV?
      #   CCDitem['Ending'] = Ending       #not found in rac 
        
    
    
        
    #    CCDitem['BC']=CCDitem['BC'][0] Bad culums ska vara en array
        CCDitem['CCDSEL']=CCDitem['CCDSEL'][0]
        CCDitem['EXPTS']=CCDitem['EXPTS'][0]
        CCDitem['EXPTSS']=CCDitem['EXPTSS'][0]
        CCDitem['FBINOV']=CCDitem['FBINOV'][0]
        CCDitem['FRAME']=CCDitem['FRAME'][0]
        CCDitem['JPEGQ']=CCDitem['JPEGQ'][0]
        CCDitem['LBLNK']=CCDitem['LBLNK'][0]
        CCDitem['NBC']=CCDitem['NBC'][0]
        CCDitem['NCOL']=CCDitem['NCOL'][0]
        CCDitem['NCSKIP']=CCDitem['NCSKIP'][0]
        CCDitem['NFLUSH']=CCDitem['NFLUSH'][0]
        CCDitem['NRBIN']=CCDitem['NRBIN'][0]        
        CCDitem['NROW']=CCDitem['NROW'][0]
        CCDitem['NRSKIP']=CCDitem['NRSKIP'][0]
        CCDitem['SID_mnemonic']=CCDitem['SID_mnemonic'][0]
        CCDitem['TBLNK']=CCDitem['TBLNK'][0]
        CCDitem['TEMP']=CCDitem['TEMP'][0]
        CCDitem['TEXPMS']=CCDitem['TEXPMS'][0]
        CCDitem['TIMING1']=CCDitem['TIMING1'][0] # Named Reserved1 in Georgis code /LM 20191115
        CCDitem['TIMING2']=CCDitem['TIMING2'][0] # Named Reserved2 in Georgis code /LM 20191115
        CCDitem['TIMING3']=CCDitem['TIMING3'][0] # Named VersionDate in Georgis code /LM 20191115
        CCDitem['VERSION']=CCDitem['VERSION'][0]
        CCDitem['WDWOV']=CCDitem['WDWOV'][0]
        CCDitem['ZERO']=CCDitem['ZERO'][0] 
    
        CCDitem['NColBinFPGA']=CCDitem['NColBinFPGA'][0]
        CCDitem['NColBinCCD']=CCDitem['NColBinCCD'][0]    
        CCDitem['DigGain']=CCDitem['DigGain'][0]
        CCDitem['TimingFlag']=CCDitem['TimingFlag'][0] 
        CCDitem['SigMode']=CCDitem['SigMode'][0]
        CCDitem['WinModeFlag']=CCDitem['WinModeFlag'][0]     
        CCDitem['WinMode']=CCDitem['WinMode'][0]  
        
    
    
        
        if int(CCDitem['CCDSEL'])==1:
            channel='IR1'
        elif int(CCDitem['CCDSEL'])==4:
            channel='IR2'
        elif int(CCDitem['CCDSEL'])==3:
            channel='IR3'
        elif int(CCDitem['CCDSEL'])==2:
            channel='IR4'
        elif int(CCDitem['CCDSEL'])==5:
            channel='UV1'
        elif int(CCDitem['CCDSEL'])==6:
            channel='UV2'
        elif int(CCDitem['CCDSEL'])==7:
            channel='NADIR'
        else:
            print('Error in CCDSEL, CCDSEL=',int(CCDitem['CCDSEL']))  
            
        
        CCDitem['channel']=channel
        
    

    return CCD_image_data

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value



def readimg(filename):

#    filename='/Users/lindamegner/MATS/retrieval/Level1/data/June2018TVAC/IMAGES/1245852147_data.npy'
    data_arr=np.fromfile(filename, dtype='uint16')#check endianess, uint16 instead of ubit16 seems to work

 
    #convert header to binary
    #header_bin = [bin(np.double(data_arr[i])) for i in range(0,12)]#check if at least 16 bits, check if indexing correct
    header_bin = np.asarray([bin(data_arr[i]) for i in range(0,12)])#this is a string 
    #change format of header_bin elements to be formatted like matlab char array
    for i in range(0,len(header_bin)):
        header_bin[i]=header_bin[i][2:].zfill(16)
    #read header
    Frame_count = int(header_bin[0],2)
    NRow = int(header_bin[1],2)
    NRowBinCCD = int(header_bin[2][10:16],2)
    NRowSkip = int(header_bin[3][8:16],2)
    NCol = int(header_bin[4],2)
    NColBinFPGA = int(header_bin[5][2:8],2)
    NColBinCCD = int(header_bin[5][8:16],2)
    NColSkip = int(header_bin[6][5:16],2)
    N_flush = int(header_bin[7],2)
    Texposure_MSB = int(header_bin[8],2)
    Texposure_LSB = int(header_bin[9],2)
    Gain = int(header_bin[10],2)
    SignalMode = Gain & 4096 
    Temperature_read = int(header_bin[11],2)

    #read image
    if len(data_arr)< NRow*(NCol+1)/(2**(NColBinFPGA)):#check for differences in python 2 and 3
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
        Ending = 'Wrong size'
    else:
        img_flag = 1
        image = np.reshape(np.double(data_arr[11+1:NRow*(NCol+1)+12]), (NCol+1,NRow))
        image = np.matrix(image).getH()
    
        #Trailer
        trailer_bin = np.asarray([bin(data_arr[i]) for i in range(NRow*(NCol+1)+12,len(data_arr))])
        Noverflow = int(trailer_bin[0],2)
        BlankLeadingValue = int(trailer_bin[1],2)
        BlankTrailingValue = int(trailer_bin[2],2)
        ZeroLevel = int(trailer_bin[3],2)
        
        Reserved1 = int(trailer_bin[4],2)
        Reserved2 = int(trailer_bin[5],2)
        
        Version = int(trailer_bin[6],2)
        VersionDate = int(trailer_bin[7],2)
        NBadCol = int(trailer_bin[8],2)
        BadCol = []
        Ending = int(trailer_bin[-1],2)
        
        if NBadCol > 0:
            BadCol = np.zeros(NBadCol)
            for k_bc in range(0,NBadCol):
                BadCol[k_bc] = int(trailer_bin[9+k_bc],2)#check if this actually works
    
    #original matlab code uses structured array, as of 20-03-2019 implementation as dictionary seems to be more useful choice
    #decision might depend on further (computational) use of data, which is so far unknown to me
    header = {}
    header['Size'] = len(data_arr)
    header['FRAME'] = Frame_count
    header['NROW'] = NRow
    header['NRBIN'] = NRowBinCCD
    header['NRSKIP'] = NRowSkip
    header['NCOL'] = NCol
    header['NColBinFPGA'] = NColBinFPGA
    header['NColBinCCD'] = NColBinCCD
    header['NCSKIP'] = NColSkip
    header['NFLUSH'] = N_flush
    header['TEXPMS'] = Texposure_LSB + Texposure_MSB*2**16
    header['DigGain'] = Gain & 15 
    header['SigMode'] = SignalMode
    header['TEMP'] = Temperature_read
    header['Noverflow'] = Noverflow # FBINOV?
    header['LBLNK'] = BlankLeadingValue
    header['TBLNK'] = BlankTrailingValue
    header['ZERO'] = ZeroLevel
    header['TIMING1'] = Reserved1
    header['TIMING2'] = Reserved2
    header['VERSION'] = Version
    header['TIMING3'] = VersionDate
    header['NC'] = NBadCol
    header['BC'] = BadCol
    header['Ending'] = Ending        
    
    
    header['CCDSEL'] = 1 #LM hardcoded - should not be used   
    header['channel'] = 'KTH test channel' 
    return image, header, img_flag


def readimage_create_CCDitem(path, file_number): #reads file from georigis stuff LM20191113

    

    
    filename = '%sF_0%02d/D_0%04d' % (path, np.floor(file_number/100),file_number)

    
#    filename='/Users/lindamegner/MATS/retrieval/Level1/data/June2018TVAC/IMAGES/1245852147_data.npy'
    data_arr=np.fromfile(filename, dtype='uint16')#check endianess, uint16 instead of ubit16 seems to work

 
    #convert header to binary
    #header_bin = [bin(np.double(data_arr[i])) for i in range(0,12)]#check if at least 16 bits, check if indexing correct
    header_bin = np.asarray([bin(data_arr[i]) for i in range(0,12)])#this is a string 
    #change format of header_bin elements to be formatted like matlab char array
    for i in range(0,len(header_bin)):
        header_bin[i]=header_bin[i][2:].zfill(16)
    #read header
    Frame_count = int(header_bin[0],2)
    NRow = int(header_bin[1],2)
    NRowBinCCD = int(header_bin[2][10:16],2)
    NRowSkip = int(header_bin[3][8:16],2)
    NCol = int(header_bin[4],2)
    NColBinFPGA = int(header_bin[5][2:8],2)
    NColBinCCD = int(header_bin[5][8:16],2)
    NColSkip = int(header_bin[6][5:16],2)
    N_flush = int(header_bin[7],2)
    Texposure_MSB = int(header_bin[8],2)
    Texposure_LSB = int(header_bin[9],2)
    Gain = int(header_bin[10],2)
    SignalMode = Gain & 4096 
    Temperature_read = int(header_bin[11],2)


    #read image
    if len(data_arr)< NRow*(NCol+1)/(2**(NColBinFPGA)):#check for differences in python 2 and 3
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
        Ending = 'Wrong size'
    else:
        img_flag = 1
        image = np.reshape(np.double(data_arr[11+1:NRow*(NCol+1)+12]), (NCol+1,NRow))
        image = np.matrix(image).getH()
    
        #Trailer
        trailer_bin = np.asarray([bin(data_arr[i]) for i in range(NRow*(NCol+1)+12,len(data_arr))])
        Noverflow = int(trailer_bin[0],2)
        BlankLeadingValue = int(trailer_bin[1],2)
        BlankTrailingValue = int(trailer_bin[2],2)
        ZeroLevel = int(trailer_bin[3],2)
        
        Reserved1 = int(trailer_bin[4],2)
        Reserved2 = int(trailer_bin[5],2)
        
        Version = int(trailer_bin[6],2)
        VersionDate = int(trailer_bin[7],2)
        NBadCol = int(trailer_bin[8],2)
        BadCol = []
        Ending = int(trailer_bin[-1],2)
        
        if NBadCol > 0:
            BadCol = np.zeros(NBadCol)
            for k_bc in range(0,NBadCol):
                BadCol[k_bc] = int(trailer_bin[9+k_bc],2)#check if this actually works
    
    #original matlab code uses structured array, as of 20-03-2019 implementation as dictionary seems to be more useful choice
    #decision might depend on further (computational) use of data, which is so far unknown to me
    CCDitem = {}
#    CCDitem['Size'] = len(data_arr)
    CCDitem['FRAME'] = Frame_count
    CCDitem['NROW'] = NRow
    CCDitem['NRBIN'] = NRowBinCCD
    CCDitem['NRSKIP'] = NRowSkip
    CCDitem['NCOL'] = NCol
    CCDitem['NColBinFPGA'] = NColBinFPGA 
    CCDitem['NColBinCCD'] = NColBinCCD
    CCDitem['NCSKIP'] = NColSkip
    CCDitem['NFLUSH'] = N_flush
    CCDitem['TEXPMS'] = Texposure_LSB + Texposure_MSB*2**16
    CCDitem['DigGain'] = Gain & 15 
    CCDitem['SigMode'] = SignalMode 
    CCDitem['TEMP'] = Temperature_read
    CCDitem['Noverflow'] = Noverflow # FBINOV?
    CCDitem['LBLNK'] = BlankLeadingValue  
    CCDitem['TBLNK'] = BlankTrailingValue
    CCDitem['ZERO'] = ZeroLevel
    CCDitem['TIMING1'] = Reserved1
    CCDitem['TIMING2'] = Reserved2 
    CCDitem['VERSION'] = Version
    CCDitem['TIMING3'] = VersionDate 
    CCDitem['NBC'] = NBadCol
    CCDitem['BC'] = BadCol
    CCDitem['Ending'] = Ending       #not found in rac 
    
    CCDitem['IMAGE']=image
    
    CCDitem['CCDSEL'] = 1 # Note that this is incorrect but hte KTH test CCD is unknown    
    CCDitem['channel'] = 'KTH test channel' 
    

    
    
    
    return CCDitem, img_flag



# Why has image values of 10 000 when raw_image has values of 400?
def readracimg(filename):
#   Linda Megner; function to read in from rac file but yield similar result 
#    as when read in by readimg . Note that this header has more info.
    
    
    
    image, metadata = imagereader.read_MATS_image(filename) 
    image=np.float64(image)
#   LM Do we want image to be float or integer? When do we want to convert it?     

    header = {}
#    header['Size'] = image.size
#   LM: Note this is NOT equivalent to what is in readim    
#    header['Size'] = len(data_arr). This variable nopt needed

  
    header['Frame_count'] = metadata['FRAME'][0]
    header['NRow'] = metadata['NROW'][0]
    header['NRowBinCCD'] = metadata['NRBIN'][0]
    header['NRowSkip'] = metadata['NRSKIP'][0]
    header['NCol'] = metadata['NCOL'][0]
    binstr_NCBIN=bin(metadata['NCBIN'][0])[2:].zfill(16)
    header['NColBinFPGA'] = int(binstr_NCBIN[2:8]) #Note that bit 0 is binstr_NCBIN[16]
    header['NColBinCCD'] = int(binstr_NCBIN[8:16])
    header['NColSkip'] = metadata['NCSKIP'][0]
    header['N_flush'] = metadata['NFLUSH'][0]
    header['Texposure'] = metadata['TEXPMS'][0]
    header['DigGain'] = metadata['GAIN'][0] & 15 
    
    header['SignalMode'] =  metadata['GAIN'][0] & 4096 #LM Selects bit 13 not 12 !  #
    header['Temperature_read'] = metadata['TEMP'][0] #LM what is this the temperature of and how do I convert it? 
#  Temperature is measured in the CRBD – readout ADC, so closer to OBC.
#It’s a 15 bit value from 2.048V (+/-0.5%) reference.
#So 4307/32768*2048 = 269 mV
#According to datasheet ADC outputs 273mV @ 25°C with 0.85 mV/°C sensitivity. So it’s approximately 20-21°C when the image was taken.
    header['Noverflow'] =  metadata['FBINOV'][0]  
    header['BlankLeadingValue'] =  metadata['LBLNK'][0]  
    header['BlankTrailingValue'] =  metadata['TBLNK'][0]  
    header['ZeroLevel'] =  metadata['ZERO'][0] 

    
    header['Reserved1'] = metadata['TIMING1'][0] #Check that this is correct by comparing old and new software ICD
    header['Reserved2'] = metadata['TIMING2'][0] #Check that this is correct by comparing old and new software ICD
    header['Version'] = metadata['VERSION'][0]
    header['VersionDate'] = metadata['TIMING3'][0] #Check that this is correct by comparing old and new software ICD
    header['NBadCol'] = metadata['NBC'][0]
    header['BadCol'] = metadata['BC']
#    header['Ending'] = 64175    NO INFO IN RAC FILE metadata 
    
#   More information not used in readimg
    header['CCDSEL'] = metadata['CCDSEL'][0]  
    header['EXPTS'] = metadata['EXPTS'][0]   
    header['EXPTSS'] = metadata['EXPTSS'][0]   
    header['JPEGQ'] = metadata['JPEGQ'][0]  
    header['SID_mnemonic'] = metadata['SID_mnemonic'][0] 
    header['WDW'] = metadata['WDW'][0]   
    header['WDWOV'] = metadata['WDWOV'][0]
    
    header['TIMIMG_FLAG'] = metadata['GAIN'][0]  & 256
    
    
#    img_flag=1 #LM is this needed? Ask Georgi
    return image, header





def readimgpath(path, file_number, plot):
    #stuff happens

    filename = '%sF_0%02d/D_0%04d' % (path, np.floor(file_number/100),file_number)

    image, header, img_flag = readimg(filename)



    if plot>0:
        #do the plotting
        mean_img=np.mean(image)
    
        plt.imshow(image, cmap='viridis', vmin=mean_img-100, vmax=mean_img+100)
        plt.title('CCD image')
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        plt.show()#equivalent to hold off
        
        print(header)
        time.sleep(0.1)
    
    return image, header


def read_txtfile(filepath):    
    dict = {}
    with open(filepath, "r") as f:
        for line in f:
            (key, val) = line.split("=")
            if key=='id':
                dict['ID'] = val[1:-2] #For some reason a space has been added before and an ewnd og line character in the end - remove these
            else:
                dict[key] = int(val)
    return dict
    
def read_txtfile_create_CCDitem(filepath):
    # function used to read txt output from image viewer
    CCDitem=read_txtfile(filepath)


# Extract variables from certain bits within the same element, see 6.4.1 Software ICD /LM 20191115               
    CCDitem['NColBinFPGA'] = int(CCDitem['NCBIN']) & (4096-256)
    CCDitem['NColBinCCD'] = int(CCDitem['NCBIN']) & 255
    del CCDitem['NCBIN']
    CCDitem['DigGain'] = int(CCDitem['GAIN']) & 15 
    CCDitem['TimingFlag'] = int(CCDitem['GAIN']) & 256
    CCDitem['SigMode'] =  int(CCDitem['GAIN']) & 4096            
    del CCDitem['GAIN']
    CCDitem['WinModeFlag']=int(CCDitem['WDW'])& 128
    CCDitem['WinMode']=int(CCDitem['WDW'])& 7
    del CCDitem['WDW']
    

    
    
    print('The following values may be incorrect:  NColBinFPGA, NColBinCCD DigGain TimingFlag SigMode inModeFlag WinMode')
    if int(CCDitem['CCDSEL'])==1:
        channel='IR1'
    elif int(CCDitem['CCDSEL'])==4:
        channel='IR2'
    elif int(CCDitem['CCDSEL'])==3:
        channel='IR3'
    elif int(CCDitem['CCDSEL'])==2:
        channel='IR4'
    elif int(CCDitem['CCDSEL'])==5:
        channel='UV1'
    elif int(CCDitem['CCDSEL'])==6:
        channel='UV2'
    elif int(CCDitem['CCDSEL'])==7:
        channel='NADIR'
    else:
        print('Error in CCDSEL, CCDSEL=',int(CCDitem['CCDSEL']))  
    CCDitem['channel']=channel
    
    return CCDitem


def get_true_image(image, header):
    #calculate true image by removing readout offset (pixel blank value) and
    #compensate for bad colums 
    
    ncolbinC=int(header['NColBinCCD'])
    if ncolbinC == 0:
        ncolbinC = 1
    
    #remove gain
    true_image = image * 2**(int(header['DigGain'])) 
    
    #bad column analysis
    n_read, n_coadd = binning_bc(int(header['NCOL'])+1, int(header['NCSKIP']), 2**int(header['NColBinFPGA']), ncolbinC, header['BC'])
    
    #go through the columns
    for j_c in range(0, int(header['NCOL'])):
        #remove blank values and readout offsets
        true_image[0:int(header['NROW']), j_c] = true_image[0:int(header['NROW']), j_c] - n_read[j_c]*(header['TBLNK']-128)-128
        
        #compensate for bad columns
        true_image[0:int(header['NROW']), j_c] = true_image[0:int(header['NROW']), j_c] * (2**int(header['NColBinFPGA'])*ncolbinC/n_coadd[j_c])
    
    return true_image

def binning_bc(Ncol, Ncolskip, NcolbinFPGA, NcolbinCCD, BadColumns):
    
    #a routine to estimate the correction factors for column binning with bad columns
    
    #n_read - array, containing the number of individually read superpixels
    #           attributing to the given superpixel
    #n_coadd - array, containing the number of co-added individual pixels
    #Input - as per ICD. BadColumns - array containing the index of bad columns
    #           (the index of first column is 0)
    
    n_read=np.zeros(Ncol)
    n_coadd=np.zeros(Ncol)
    
    col_index=Ncolskip
    
    for j_col in range(0,Ncol):
        for j_FPGA in range(0,NcolbinFPGA):
            continuous=0
            for j_CCD in range(0,NcolbinCCD):
                if col_index in BadColumns:
                    if continuous == 1:
                        n_read[j_col]=n_read[j_col]+1
                    continuous=0
                else:
                    continuous=1
                    n_coadd[j_col]=n_coadd[j_col]+1
                
                col_index=col_index+1
            
            if continuous == 1:
                n_read[j_col]=n_read[j_col]+1
    
    return n_read, n_coadd


def desmear_true_image(image, header):
    
    nrow = int(header['NROW'])
    ncol = int(header['NCOL']) + 1
    
    #calculate extra time per row
    T_row_extra, T_delay = calculate_time_per_row(header)
    T_exposure = int(header['TEXPMS'])/1000#check for results when shifting from python 2 to 3
    
    TotTime=0
    for irow in range(1,nrow):
        for krow in range(0,irow):
            image[irow,0:ncol]=image[irow,0:ncol] - image[krow,0:ncol]*(T_row_extra/T_exposure)
            TotTime=TotTime+T_row_extra
           
    #row 0 here is the first row to read out from the chip

    return image

def calculate_time_per_row(header):
    
    #this function provides some useful timing data for the CCD readout
    
    #Note that minor "transition" states may have been omitted resulting in 
    #somewhat shorter readout times (<0.1%).
    
    #Default timing setting is_
    #ccd_r_timing <= x"A4030206141D"&x"010303090313"
    
    #All pixel timing setting is the final count of a counter that starts at 0,
    #so the number of clock cycles exceeds the setting by 1
    
    #image parameters
    ncol=int(header['NCOL'])+1
    ncolbinC=int(header['NColBinCCD'])
    if ncolbinC == 0:
        ncolbinC = 1
    ncolbinF=2**int(header['NColBinFPGA'])
    
    nrow=int(header['NROW'])
    nrowbin=int(header['NRBIN'])
    if nrowbin == 0:
        nrowbin = 1
    nrowskip=int(header['NRSKIP'])
    
    n_flush=int(header['NFLUSH'])
    
    #timing settings
    full_timing = 1 #TODO <-- meaning?
    
    #full pixel readout timing n#TODO discuss this with OM,  LMc these are default values change these when the header contians this infromation
    
    time0 = 1 + 19 # x13%TODO
    time1 = 1 +  3 # x03%TODO
    time2 = 1 +  9 # x09%TODO
    time3 = 1 +  3 # x03%TODO
    time4 = 1 +  3 # x03%TODO
    time_ovl = 1 + 1 # x01%TODO
    
    # fast pixel readout timing
    timefast  = 1 + 2 # x02%TODO
    timefastr = 1 + 3 # x03%TODO
    
    #row shift timing
    row_step = 1 + 164 # xA4%TODO
    
    clock_period = 30.517 #master clock period, ns 32.768 MHz
    
    #there is one extra clock cycle, effectively adding to time 0
    Time_pixel_full = (1+ time0 + time1 + time2 + time3 + time4 + 3*time_ovl)*clock_period
    
    # this is the fast timing pixel period
    Time_pixel_fast = (1+ 4*timefast + 3*time_ovl + timefastr)*clock_period
    
    #here we calculate the number of fast and slow pixels
    #NOTE: the effect of bad pixels is disregarded here
    
    if full_timing == 1:
        n_pixels_full = 2148
        n_pixels_fast = 0
    else:
        if ncolbinC < 2: #no CCD binning
            n_pixels_full = ncol * ncolbinF
        else: #there are two "slow" pixels for one superpixel to be read out
            n_pixels_full = 2*ncol *ncolbinF
        n_pixels_fast = 2148 - n_pixels_full
    
    
    #time to read out one row
    T_row_read = n_pixels_full*Time_pixel_full + n_pixels_fast*Time_pixel_fast
    
    # shift time of a single row
    T_row_shift = (64 + row_step *10)*clock_period
    
    #time of the exposure start delay from the start_exp signal # n_flush=1023
    T_delay = T_row_shift * n_flush
    
    #total time of the readout
    T_readout = T_row_read*(nrow+nrowskip+1) + T_row_shift*(1+nrowbin*nrow)
    
    
    #"smearing time"
    #(this is the time that any pixel collects electrons in a wrong row, during the shifting.)
    #For smearing correction, this is the "extra exposure time" for each of the rows.
    
    T_row_extra = (T_row_read + T_row_shift*nrowbin) / 1e9    
    
    return T_row_extra, T_delay

def compare_image(image1, image2, header):
    
    #this is a function to compare two images of the same size
    #one comparison is a linear fit of columns, the other comparison is a linear fit
    #of rows, the third is a linear fit of the whole image
    
    sz1=image1.shape
    sz2=image2.shape
    
    if sz1[0] != sz2[0] or sz1[1] != sz2[1]:
        print('sizes of input images do not match')
    
    nrow=sz1[0]
    ncol=sz1[1]
    
    nrowskip = int(header['NRSKIP'])
    ncolskip = int(header['NCSKIP'])
    
    nrowbin = int(header['NRBIN'])
    ncolbinC = int(header['NCBIN'])
    ncolbinF = 2**int(header['NColBinFPGA'])
    
    if nrowskip + nrowbin*nrow > 511:
        nrow = np.floor((511-nrowskip)/nrowbin)
        
    if ncolskip + ncolbinC*ncolbinF*ncol > 2047:
        nrow = np.floor((2047-ncolskip)/(ncolbinC*ncolbinF))
    print(nrow,image1.shape)        
    image1 = image1[0:nrow-1, 0:ncol-1]
    image2 = image2[0:nrow-1, 0:ncol-1]
    
    r_scl=np.zeros(nrow)
    r_off=np.zeros(nrow)
    r_std=np.zeros(nrow)
    
    for jj in range(0,nrow-1):
        x=np.concatenate((np.ones((ncol-1,1)), np.expand_dims(image1[jj,].conj().transpose(), axis=1)), axis=1)#-1 to adjust to python indexing?
        y=image2[jj,].conj().transpose()
        bb, ab, aa, cc = np.linalg.lstsq(x,y)
        
        ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
        #ft=np.multiply(x[:,1]*bb[1]) + bb[0]

        adf=np.abs(np.squeeze(y)-np.squeeze(ft))
        sigma=np.std(np.squeeze(y)-np.squeeze(ft))

        inside = np.where(adf < 2*sigma)
        bb, ab, aa, cc = np.linalg.lstsq(x[inside[1],], y[inside[1]])
        
        ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
        
        r_scl[jj]=bb[1]
        r_off[jj]=bb[0]
        r_std[jj]=np.std(y[0]-ft[0])
        
    c_scl=np.zeros(nrow)
    c_off=np.zeros(nrow)
    c_std=np.zeros(nrow)
   
    for jj in range(0,ncol-1):
        
        x=np.concatenate((np.ones((nrow-1,1)), np.expand_dims(image1[:,jj], axis=1)), axis=1)
        y=image2[:,jj]
        bb, ab, aa, cc = np.linalg.lstsq(x,y)
        
        ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
        
        adf=np.abs(np.squeeze(y)-np.squeeze(ft))
        sigma=np.std(np.squeeze(y)-np.squeeze(ft))

        inside = np.where(adf < 2*sigma)
        bb, ab, aa, cc = np.linalg.lstsq(x[inside[1],], y[inside[1]])
            
        ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
        
        c_scl[jj]=bb[1]
        c_off[jj]=bb[0]
        c_std[jj]=np.std(y[0]-ft[0])
    
    nsz=(nrow-1)*(ncol-1)
    la_1=np.reshape(image1, (nsz,1))
    la_2=np.reshape(image2, (nsz,1))
    
    x=np.concatenate((np.ones((nsz,1)),la_1), axis=1)
    y=la_2
    bb, ab, aa, cc = np.linalg.lstsq(x,y)
    
    ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
    
    adf=np.abs(np.squeeze(y)-np.squeeze(ft))
    sigma=np.std(np.squeeze(y)-np.squeeze(ft))
    
    inside = np.where(adf < 2*sigma)
    bb, ab, aa, cc = np.linalg.lstsq(x[inside[1],], y[inside[1]])
    
        
    ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
    
    t_off=bb[0]
    t_scl=bb[1]
    t_std=np.std(y[0]-ft[0])
    
    rows=0
    
    return t_off, t_scl, t_std

def compensate_bad_columns(image, header):
    #LM 200127 This does not need to be used since it is already done in the OBC says Georgi.
    
    #this is a function to compensate bad columns if in the image
    
    ncol=int(header['NCOL'])+1
    nrow=int(header['NROW'])
    
    ncolskip=int(header['NCSKIP'])
    
    ncolbinC=int(header['NCBIN'])
    ncolbinF=2**int(header['NColBinFPGA'])
    
    #change to Leading if Trailing does not work properly
    blank=int(header['TBLNK'])
    
    gain=2**(int(header['DigGain']))
    
    if ncolbinC == 0: #no binning means binning of one
        ncolbinC=1
    
    if ncolbinF==0: #no binning means binning of one
        ncolbinF=1
    
    #bad column analysis
    
    n_read, n_coadd = binning_bc(ncol, ncolskip, ncolbinF, ncolbinC, np.asarray(header['BC']))
    
    if ncolbinC>1:
        for j_c in range(0,ncol):
            if ncolbinC*ncolbinF != n_coadd[j_c]:
                #remove gain adjustment
                image[0:nrow-1,j_c] = image[0:nrow-1,j_c]*gain
                
                #remove added superpixel value due to bad columns and read out offset
                image[0:nrow-1,j_c] = image[0:nrow-1,j_c] - n_read[j_c]*(blank-128) -128
                
                #multiply by number of binned column to actual number readout ratio
                image[0:nrow-1,j_c] = image[0:nrow-1,j_c] * ((ncolbinC*ncolbinF)/n_coadd[j_c])
                
                #add number of FPGA binned
                image[0:nrow-1,j_c] = image[0:nrow-1,j_c] + ncolbinF*(blank-128) + 128
                
                #add gain adjustment back
                image[0:nrow-1,j_c] = image[0:nrow-1,j_c]/gain
                
                print('Col: ',j_c,', n_read: ',n_read[j_c],', n_coadd: ',n_coadd[j_c],', binned pixels: ',ncolbinC*ncolbinF)
    
    return image

#
#def get_true_image_from_compensated(image, header):
#    
#    #calculate true image by removing readout offset, pixel blank value and
#    #normalising the signal level according to readout time
#    
#    #remove gain
#    true_image = image * 2**(int(header['DigGain']) & 255)
#    
#    for j_c in range(0,int(header['NCol'])):
#        true_image[0:header['NRow'], j_c] = ( true_image[0:header['NRow'],j_c] - 
#                  2**header['NColBinFPGA'] * (header['BlankTrailingValue']-128) - 128 )
#    
#    
#    return true_image




class CCD:
    def __init__(self, channel):
        self.channel=channel
        if channel=='IR1':
            CCDID=16
        elif channel=='IR2':
            CCDID=17
        elif channel=='IR3':
            CCDID=18
        elif channel=='IR4':
            CCDID=19
        elif channel=='NADIR':
            CCDID=20
        elif channel=='UV1':
            CCDID=21
        elif channel=='UV2':
            CCDID=22            
        elif channel=='KTH test channel':
            CCDID=16 
        
        darkdir='/Users/lindamegner/MATS/retrieval/Calibration/FM_calibration_at_KTH/FM_CCD_DC_calibration_data_reduced/'
        filename= darkdir+ 'FM0' +str(CCDID) +'_CCD_DC_calibration_DATA_reduced.mat'
        mat = scipy.io.loadmat(filename)

        self.dc_zero_avr_HSM=mat['dc_zero_avr_HSM']
        self.dc_zero_std_HSM=mat['dc_zero_std_HSM']
        self.dc_zero_avr_LSM=mat['dc_zero_avr_LSM']
        self.dc_zero_std_LSM=mat['dc_zero_std_LSM']
        
        self.image_HSM=mat['image_HSM']
        self.image_LSM=mat['image_LSM']
        
        
        self.ro_avr_HSM=mat['ro_avr_HSM']
        self.ro_std_HSM=mat['ro_std_HSM']
        self.alpha_avr_HSM=mat['alpha_avr_HSM']
        self.alpha_std_HSM=mat['alpha_std_HSM']
        
        self.ro_avr_LSM=mat['ro_avr_LSM']
        self.ro_std_LSM=mat['ro_std_LSM']
        self.alpha_avr_LSM=mat['alpha_avr_LSM']
        self.alpha_std_LSM=mat['alpha_std_LSM']
        
        self.log_a_avr_HSM=mat['log_a_avr_HSM']
        self.log_a_std_HSM=mat['log_a_std_HSM']
        self.log_b_avr_HSM=mat['log_b_avr_HSM']
        self.log_b_std_HSM=mat['log_b_std_HSM']
        
        self.log_a_avr_LSM=mat['log_a_avr_LSM']
        self.log_a_std_LSM=mat['log_a_std_LSM']
        self.log_b_avr_LSM=mat['log_b_avr_LSM']
        self.log_b_std_LSM=mat['log_b_std_LSM']
        
        self.hot_pix=np.where(self.image_HSM>=0.8*np.max(self.image_HSM))

        if   (self.channel=='UV1' or self.channel=='UV2'):
            self.ampcorrection=3/2 #Amplification hack - check with Gabriel how to to properly
        else:
            self.ampcorrection=1
                  
    def darkcurrent(self, T, mode): #electrons/s
        if mode == 0: 
            darkcurrent=10**(self.log_a_avr_HSM*T+self.log_b_avr_HSM)
        elif mode == 1:
            darkcurrent=10**(self.log_a_avr_LSM*T+self.log_b_avr_LSM)
        else :
            print('Undefined mode')
        return darkcurrent
    
    def ro_avr(self, mode):
        if mode == 0: 
            ro_avr=self.ro_avr_HSM
        elif mode == 1:
            ro_avr=self.ro_avr_LSM
        else :
            print('Undefined mode')
        return ro_avr
    
    def alpha_avr(self, mode): #electrons/LSB
        if mode == 0: 
            alpha_avr=self.alpha_avr_HSM
        elif mode == 1:
            alpha_avr=self.alpha_avr_LSM
        else :
            print('Undefined mode')
        return alpha_avr   
    
