#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:32:55 2019

@author: lindamegner 


This is the man MATS level 1 datachain. It does the following:
    1)removes bias (as measured by blank pixels)
    2)compensates for bad columns (ie attempts to recreate true image from the compensated one in MATS
    payload OBC) 
    3) desmears
    4)removes dark current
    5)removes flat field
    6)converts to photons per binned pixel
    
This chain shouls output one image with unit photons per pixal 
and one image that gives the error per pixel also in photons per pixel
    


TODOs:
    error handling
    window mode
    check ampcorrection with gabril/georgi

"""


import numpy as np
import matplotlib.pyplot as plt
from L1_calibration_functions import get_true_image, desmear_true_image, CCD, read_CCDitem, readimage_create_CCDitem, readimageviewpic
from get_temperature import create_temperature_info_array

import datetime 
import json
import sys
# Insert path: insert 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/lindamegner/MATS/retrieval/Level0/MATS-L0-processing-master')



def plot(image,title,darkarea):
    plt.figure()
    mean_img=image.mean()
    mean_at_dark=np.mean(image[darkarea[0]:darkarea[1],darkarea[2]:darkarea[3]])
    print('mean at dark=', mean_at_dark)
    std=image.std()
    p0= plt.imshow(image, aspect='auto',origin='lower', vmin=mean_img-2*std, vmax=mean_img+2*std)
    plt.title(title+ ' mean= '+str(mean_img))
    plt.xlabel('Pixels')    
    plt.ylabel('Pixels')
    plt.colorbar()
    plt.text(150,100,'mean at dark= '+ str(mean_at_dark))

    return  p0


#################################################
# Read in the data                              #
#################################################

read_from=1 # 0 is from KTH images, 1 is fron rac file, 2 is from image viewer


if read_from==0: #Read KTH files
    CCDitem, flag =readimage_create_CCDitem('/Users/lindamegner/MATS/retrieval/Level1/data/2019-02-08 rand6/', 1)
elif read_from==1: #Read from rac file new version as of November 2019 
    rac_image_json_file='images.json'
    rac_packets_json_file='packets.json'     
#    rac_image_json_file='rac20190818-152721/images.json'
    rac_sub_dir='rac20191106testme/'
    retrieval_dir='/Users/lindamegner/MATS/retrieval/Level0/MATS-L0-processing-master/'
    CCDitem=read_CCDitem(rac_image_json_file,rac_sub_dir,7,retrieval_dir)
elif read_from==2: #read image and textfile created by image viewer
    rawflag=1
    dirname='/Users/lindamegner/MATS/retrieval/Calibration/FM_tests_after_glue/20191106/'
    picnr=14976
    CCDitem=readimageviewpic(dirname,picnr,rawflag)


# Create temperature information array. 
#This should be done once for every json packet
if read_from==1:    
    temperaturedata, relativetimedata=create_temperature_info_array(retrieval_dir+rac_sub_dir+rac_packets_json_file)

   

#  Hack to have no compensation for bad colums at the time. TODO later.
CCDitem['NBC']=0
CCDitem['BC']=np.array([]) 


# Read time stamp.  Possibly move to CCDitem
epoch=datetime.datetime(1980,1,6)
try:
    reltime=int(CCDitem['EXPTS'])+int(CCDitem['EXPTSS'])/2**16
    timestamp=epoch+datetime.timedelta(0,reltime)
    print(timestamp)
except: 
    print('No time info in CCDitem')


#Find the temperature of the CCDs. If not read from rac set the temperature.
if read_from==1:        
    #find the closest time when heater settings have been recorded. Could be changed to interpolate.
    ind = (np.abs(relativetimedata-reltime)).argmin()
    print(relativetimedata[ind],' reltime ', reltime)   
    HTR1A=temperaturedata[ind,0]
    HTR1B=temperaturedata[ind,1]
    HTR2A=temperaturedata[ind,2]
    HTR2B=temperaturedata[ind,3]
    HTR8A=temperaturedata[ind,4]
    HTR8B=temperaturedata[ind,5]
    temperature=HTR8B    
elif read_from==0: #Take temperature from ADC
    #Check ADC temperature. This will not be part of the calibration routine but is used as a sanity test.  
    #273mV @ 25°C with 0.85 mV/°C
    ADC_temp_in_mV=int(CCDitem['TEMP'])/32768*2048 
    ADC_temp_in_degreeC=1./0.85*ADC_temp_in_mV-296
    temperature=ADC_temp_in_degreeC #Change this to read temperature sensors from rac file
    #temperature=-18 #-18C is measured at TVAC tests in August 2019    
elif read_from==2:    
    temperature=22 # set to room temperature if no rac file
    print('Warning: No temperature infromation. Temperature is set to 22 C which will affect dark current reduction')

#Create a CCD class member that holds information specific for that particular CCDcamera
CCD=CCD(CCDitem['channel']) #TODO: Remember to clear out the non used bits of CCD /Linda



#################################################
#       L1 calibration steps                    #
#################################################


#Step0 Compensate for window mode. This should be done first, because it is done in Mikaels software
try: 
    if (CCDitem['WinMode']) <=4:
        winfactor=2**CCDitem['WinMode']
    else:        
        winfactor=1       # Check that this is what you want!  
except:
    winfactor=1
    print('no Win mode info in heater - set to 1')
image_lsb=winfactor*CCDitem['IMAGE']     

     
# Step 1 and 2: Remove bias and compensate form bad columns, image still in LSB
image_bias_sub = get_true_image(image_lsb, CCDitem)


# Step 4: Desmear

image_desmeared = desmear_true_image(image_bias_sub.copy(), CCDitem)



# Step 5 Remove dark current
# TBD: Adjust so that the dark current is different for every pixel when Gabriels new data comes. 
# TBD: The temperature needs to be decided in a better way then taken from the ADC as below.
# Either read from rac files of temperature sensors or estimated from the top of the image




print('temperature',temperature)    

totdarkcurrent=CCD.darkcurrent(temperature,int(CCDitem['SigMode']))*int(CCDitem['TEXPMS'])/1000. # tot dark current in electrons
totbinpix=int(CCDitem['NColBinCCD'])*2**int(CCDitem['NColBinFPGA']) #Note that the numbers are described in differnt ways see table 69 in Software iCD
image_dark_sub=image_desmeared-totbinpix*CCD.ampcorrection*totdarkcurrent/CCD.alpha_avr(int(CCDitem['SigMode']))


# Step 6 Remove flat field of the particular CCD. TBD.

# Step 7 Transform from LSB to electrons and then to photons. TBD.




#################################################
# Plotting the results                          #
#################################################

if read_from==1 or read_from==2:  
    darkarea=[150,250,100,400]
else:
    darkarea=[50,74,50,74]
    
f1=plot(image_lsb,CCDitem['channel']+' image raw', darkarea)
f2=plot(image_bias_sub,CCDitem['channel']+' image_bias_sub', darkarea)
f3=plot(image_desmeared,CCDitem['channel']+' image_desmeared', darkarea)
f4=plot(image_dark_sub,CCDitem['channel']+' image_dark_sub', darkarea)      
