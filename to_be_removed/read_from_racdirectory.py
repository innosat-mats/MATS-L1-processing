#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:17:10 2020

Read in all image files in rac directory.
@author: lindamegner
"""

import matplotlib.pyplot as plt
from read_in_functions import read_CCDitems
from LindasCalibrationFunctions import plotCCDitem


pathdir = "/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Flatfields/Diffusor/DiffusorFlatTests/"
rac_dir = pathdir + "RacFiles_out/"


CCDitems = read_CCDitems(rac_dir)


maxplot = 4
fig, ax = plt.subplots(maxplot, 1)
for i, CCDitem in enumerate(CCDitems[:maxplot]):
    plotCCDitem(CCDitem, fig, ax[i], CCDitem["channel"])

