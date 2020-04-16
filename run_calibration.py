#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:19:08 2020

@author: lindamegner
"""


from L1_calibration_functions import read_CCDitems, readimage_create_CCDitem, readimageviewpic, readimageviewpics
from get_temperature import create_temperature_info_array, add_temperature_info
from L1_calibrate import L1_calibrate
import datetime 

import numpy as np
import matplotlib.pyplot as plt

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


# def read_images(filename,filepath,read_from):

#     #################################################
#     # Read in the data                            #
#     #################################################
    
#      # 0 read_from=0 is from KTH images, 1 is fron rac file, 2 is from image viewer
    
#     if read_from==0: #Read KTH files
#         CCDitem, flag =readimage_create_CCDitem('/Users/lindamegner/MATS/retrieval/Level1/data/2019-02-08 rand6/', 1)
#     elif read_from==1: #Read from rac file new version as of November 2019 
#         rac_image_json_file='rac20191106testme/images.json'
#         rac_packets_json_file='rac20191106testme/packets.json'    
#     #    rac_image_json_file='rac20190818-152721/images.json'
#         rac_image_json_dir='/Users/lindamegner/MATS/retrieval/Level0/MATS-L0-processing-master/'
#         CCDitem=read_CCDitems(rac_image_json_file,rac_image_json_dir)
#     elif read_from==2: #read image and textfile created by image viewer
#         rawflag=1
#         dirname='/Users/lindamegner/MATS/retrieval/Calibration/FM_tests_after_glue/20191106/'
#         picnr=14976
#         CCDitem=readimageviewpic(dirname,picnr,rawflag)
        
#     return CCDitem





read_from=2
epoch=datetime.datetime(1980,1,6)

if read_from==1:  
    #Read in all images in a json file   
    rac_image_json_dir='/Users/lindamegner/MATS/retrieval/Level0/MATS-L0-processing-master/'    
    rac_image_json_file='images.json'
    rac_packets_json_file='packets.json'  
    rac_sub_dir='rac20191106testme/'
 
    CCDitems=read_CCDitems(rac_image_json_file,rac_sub_dir,rac_image_json_dir)
# Create temperature information array. 
#This should be done once for every json packet
    temperaturedata, relativetimedata=create_temperature_info_array(rac_image_json_dir+rac_sub_dir+rac_packets_json_file)
elif read_from==2:
     rawflag=1
     picdirname='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Flatfields/20200322_flatfields_MISU'
     CCDitems=readimageviewpics(picdirname)
# Set non used values to be able to run add_temperature_info . TODO do this in a better way.
     temperaturedata=999
     relativetimedata=999


for CCDitem in CCDitems:
    CCDitem['reltime']=int(CCDitem['EXPTS'])+int(CCDitem['EXPTSS'])/2**16 
    CCDitem['read_from']=read_from
    CCDitem=add_temperature_info(CCDitem,temperaturedata,relativetimedata)
    timestamp=epoch+datetime.timedelta(0,CCDitem['reltime'])
    print(timestamp)
    print(CCDitem['temperature'])
    
    
for CCDitem in CCDitems[4:6]:    

    image_lsb,image_bias_sub,image_desmeared, image_dark_sub =L1_calibrate(CCDitem)
    



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