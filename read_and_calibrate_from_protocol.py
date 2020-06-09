#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:33:31 2020

@author: lindamegner
"""



from LindasCalibrationFunctions import  plotCCDitem
import numpy as np

from LindasCalibrationFunctions import plot_CCDimage 
from read_in_functions import readprotocol, read_all_files_in_protocol, add_temperature_info_to_CCDitems
import matplotlib.pyplot as plt
from L1_calibrate import L1_calibrate

directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/TemperatureTest/DarkMeas_20200424/'
protocol='protocol.txt'

directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Flatfields/Diffusor/DiffusorFlatTests/'
#protocol='ABOUT.txt'

protocol='ProtocolList1.txt'

read_from='rac'  
df_protocol=readprotocol(directory+protocol)

df_bright=df_protocol[df_protocol.DarkBright=='D']
CCDitems=read_all_files_in_protocol(df_bright, read_from,directory)
CCDitems=add_temperature_info_to_CCDitems(CCDitems,read_from,directory)







calibrate=True

for CCDitem in CCDitems[0:4]:
    
    if calibrate:
    
        image_lsb,image_bias_sub,image_desmeared, image_dark_sub =L1_calibrate(CCDitem)
  
        fig,ax=plt.subplots(4,1)
        plot_CCDimage(image_lsb,fig, ax[0], 'Original LSB')    
        plot_CCDimage(image_bias_sub,fig, ax[1], 'Bias subtracted')  
        plot_CCDimage(image_desmeared,fig, ax[2],' Desmeared LSB')  
        plot_CCDimage(image_dark_sub,fig, ax[3], ' Dark current subtracted LSB')          
        fig.suptitle(CCDitem['channel'])

    else:    
    
        fig=plt.figure()
        ax=fig.gca()
        plotCCDitem(CCDitem,fig, ax[0])   
      