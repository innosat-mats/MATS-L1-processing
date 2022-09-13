#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t
Created on Mon Aug 23 15:06:42 2021

@author: lindamegner


This script produced morfed images between the flatfield without baffle taken at 0 degree temperatre at MISU and that with baffle taken at 20C at OHB.
"""

import numpy as np
from PIL import Image
from mats_l1_processing.L1_calibration_functions import (read_flatfield,CCD)
from mats_l1_processing.LindasCalibrationFunctions import plot_CCDimage,read_all_files_in_protocol    
from mats_l1_processing.read_in_functions import readprotocol 
import matplotlib.pyplot as plt
#from scipy import signal
from scipy import ndimage
from scipy.signal import spline_filter

from database_generation import flatfield as flatfield
from pathlib import Path
import toml
from mats_l1_processing.instrument import Instrument
# file=open("test.txt","r")

# print("read function: ")
# print(file.read())
# print()



calibration_file='/Users/lindamegner/MATS/retrieval/git/MATS-L1-processing/scripts/calibration_data_linda.toml'

channels=['IR1','IR2','IR3','IR4','UV1','UV2']#,'NADIR' ]

sigmodes=['HSM','LSM']

calibration_data = toml.load(calibration_file)
instrument = Instrument(calibration_file)


for channel in channels:
    for sigmode in sigmodes:
        
        flatfield_morphed=flatfield.make_flatfield(channel, sigmode,instrument, calibration_data)
        Path("output").mkdir(parents=True, exist_ok=True)
        np.savetxt('output/flatfield_'+channel+'_'+sigmode+'.csv', flatfield_morphed)
        np.save('output/flatfield_'+channel+'_'+sigmode+'.npy', flatfield_morphed)





