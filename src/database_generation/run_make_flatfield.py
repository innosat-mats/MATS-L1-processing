#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:06:42 2021

@author: lindamegner


This script produced morfed images between the flatfield without baffle taken at 0 degree temperatre at MISU and that with baffle taken at 20C at OHB.
"""
#%%
import numpy as np
from database_generation import flatfield as flatfield
from pathlib import Path
import pickle

#import sys
#sys.path.append('/Users/lindamegner/MATS/MATS-retrieval/MATS-L1-processing/src/database_generation')

calibration_file='/Users/lindamegner/MATS/MATS-retrieval/MATS-analysis/Linda/calibration_data_linda.toml'


channels=['IR1','IR2','IR3','IR4','UV1','UV2']#,'NADIR' ]



for channel in channels:
     #Note, only using HSM mode images for making flatfield now LM230925
    flatfield_morphed=flatfield.make_flatfield(channel, calibration_file, plotresult=True, plotallplots=True)
    Path("output").mkdir(parents=True, exist_ok=True)
    np.savetxt('output/flatfield_'+channel+'_HSM.csv', flatfield_morphed)
    np.save('output/flatfield_'+channel+'_HSM.npy', flatfield_morphed)
    #pickle the flatfields
    #pickle.dump(flatfield_morphed, open('output/flatfield_'+channel+'_HSM.pkl', 'wb'))



#
    


# %%
