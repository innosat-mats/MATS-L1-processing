#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:06:42 2021

@author: lindamegner


This script produced morfed images between the flatfield without baffle taken at 0 degree temperatre at MISU and that with baffle taken at 20C at OHB.
"""
#%%
import numpy as np
from PIL import Image
from mats_l1_processing.experimental_utils import plot_CCDimage,read_all_files_in_protocol    
from mats_l1_processing.experimental_utils import readprotocol 
import matplotlib.pyplot as plt
#from scipy import signal
from scipy import ndimage
from scipy.signal import spline_filter

from database_generation import flatfield as flatfield
from pathlib import Path
import toml
from mats_l1_processing.instrument import Instrument


calibration_file='/Users/lindamegner/MATS/retrieval/git/MATS-L1-processing/scripts/calibration_data_linda.toml'

channels=['IR1','IR2','IR3','IR4','UV1','UV2']#,'NADIR' ]

sigmodes=['HSM','LSM']

calibration_data = toml.load(calibration_file)
instrument = Instrument(calibration_file)

#%%


channels=['IR1','IR2','IR3','IR4','UV1','UV2']
sigmode='HSM'
fig_coef, ax_coef=plt.subplots(6,1, figsize=(10,14))
for ind, channel in enumerate(channels):

    axnr=ax_coef[ind]
    coef_wo, coef_w= flatfield.make_flatfield_lin_coef(channel, sigmode, calibration_file, ax=axnr)
    axnr.set_title(channel)
    col=axnr.lines[-1].get_color()
    x=np.arange(0, 2047)
    poly1d_fn = np.poly1d(coef_wo)
    axnr.plot(poly1d_fn(x), 'b')
    print(coef_wo)
    poly1d_fn = np.poly1d(coef_w)
    axnr.plot(poly1d_fn(x), 'r')
    axnr.set_xlim([400, 1600])
    axnr.set_ylim([0.97, 1.03])

    axnr.text(1100, 1.02,'coef_wo :'+str(coef_wo) )
    axnr.text(1100, 0.98,'coef_w :'+str(coef_w) )
    print(coef_w)
    plt.tight_layout()

#%%


for channel in channels:
    for sigmode in sigmodes:
        
        flatfield_morphed=flatfield.make_flatfield(channel, sigmode,calibration_file,plot=True)
        Path("output").mkdir(parents=True, exist_ok=True)
        np.savetxt('output/flatfield_'+channel+'_'+sigmode+'.csv', flatfield_morphed)
        np.save('output/flatfield_'+channel+'_'+sigmode+'.npy', flatfield_morphed)


# %%
