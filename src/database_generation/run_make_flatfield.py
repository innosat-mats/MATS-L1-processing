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
from database_generation.experimental_utils import plot_CCDimage

#import sys
#sys.path.append('/Users/lindamegner/MATS/MATS-retrieval/MATS-L1-processing/src/database_generation')

calibration_file='/Users/lindamegner/MATS/MATS-retrieval/MATS-analysis/Linda/calibration_data_linda.toml'


channels=['IR1','IR2','IR3','IR4','UV1','UV2']#,'NADIR' ]



for channel in channels:
     #Note, only using HSM mode images for making flatfield now LM230925
    flatfield_morphed, flatfield_wo_baffle_err, baffle_scalefield=flatfield.make_flatfield(channel, calibration_file, plotresult=True, plotallplots=True)
    Path("output").mkdir(parents=True, exist_ok=True)
    np.savetxt('output/flatfield_'+channel+'_HSM.csv', flatfield_morphed)
    np.save('output/flatfield_'+channel+'_HSM.npy', flatfield_morphed)
    # The error measured as the standard deviation of three images divided by square root of 3
    np.savetxt('output/flatfield_wo_baffle_err_scaled_'+channel+'_HSM.csv', flatfield_wo_baffle_err)
    np.save('output/flatfield_wo_baffle_err_'+channel+'_HSM.npy', flatfield_wo_baffle_err)
    # The baffle scalefield, where 1 means no effect, ie the flatfield is the 
    # same as the one without baffle. 
    np.savetxt('output/baffle_scalefield_'+channel+'_HSM.csv', baffle_scalefield)
    np.save('output/baffle_scalefield_'+channel+'_HSM.npy', baffle_scalefield)


    plot_CCDimage(flatfield_morphed, title='Flatfield '+channel)
    plot_CCDimage(flatfield_wo_baffle_err, title='Flatfield wo error '+channel)
    plot_CCDimage(baffle_scalefield, title='Baffe scalefield '+channel)



    #pickle the flatfields
    #pickle.dump(flatfield_morphed, open('output/flatfield_'+channel+'_HSM.pkl', 'wb'))




# %%
