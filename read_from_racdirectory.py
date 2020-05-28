#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:17:10 2020

Test to read one json image 
@author: lindamegner
"""





import matplotlib.pyplot as plt    
from read_in_functions import read_CCDitemsx 
from LindasCalibrationFunctions import plotCCDitem, plot_CCDimage


pathdir='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/27052020_nadir_func/'
rac_dir=pathdir+'OldRacOutput/'

CCDitems=read_CCDitemsx(rac_dir, pathdir)    

 

maxplot=4
fig, ax=plt.subplots(maxplot,1)
for i, CCDitem in enumerate(CCDitems[:maxplot]):
    plotCCDitem(CCDitem,fig,ax[i],CCDitem['channel'])


