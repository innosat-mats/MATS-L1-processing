#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:33:31 2020

@author: lindamegner
"""



from LindasCalibrationFunctions import  plotCCDitem



from read_in_functions import read_all_files_in_directory
import matplotlib.pyplot as plt
from L1_calibrate import calibrate_all_items


#directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/27052020_nadir_func/'
#directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/NadirTests/27052020_nadir_lightleakage/'

directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Flatfields/20200511_temperature_dependence/'
read_from='rac'  
CCDitems=read_all_files_in_directory(read_from,directory)



calibrate=False

if calibrate:

    calibrate_all_items(CCDitems[:2], plot=True)


else:    
    
    for CCDitem in CCDitems[:4]:
        fig=plt.figure()
        ax=fig.gca()
        plotCCDitem(CCDitem,fig, ax, title=CCDitem['channel'])
        #ax.text(200,450,CCDitem['channel'])