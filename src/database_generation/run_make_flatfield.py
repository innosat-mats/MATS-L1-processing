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
from matplotlib import pyplot as plt
from database_generation.flatfield import scale_field
#import sys
#sys.path.append('/Users/lindamegner/MATS/MATS-retrieval/MATS-L1-processing/src/database_generation')

calibration_file='/Users/lindamegner/MATS/MATS-retrieval/MATS-analysis/Linda/calibration_data_linda.toml'


channels=['IR1','IR2','IR3','IR4','UV1','UV2']#,'NADIR' ]


flatfields_errors = {}
for channel in channels:
    fig, ax=plt.subplots(2,1, figsize=(8,5))

    #Note, only using HSM mode images for making flatfield now LM230925
    flatfield_scaled, flatfield_err=flatfield.make_flatfield(channel, calibration_file, plotresult=True, plotallplots=True)
    flatfields_errors[channel]=flatfield_err
    Path("output").mkdir(parents=True, exist_ok=True)
    np.savetxt('output/flatfield_'+channel+'_HSM.csv', flatfield_scaled)
    np.save('output/flatfield_'+channel+'_HSM.npy', flatfield_scaled)
    
    np.savetxt('output/flatfield_err_'+channel+'_HSM.csv', flatfield_err)
    np.save('output/flatfield_err_'+channel+'_HSM.npy', flatfield_err)


    plot_CCDimage(flatfield_scaled, fig=fig, axis=ax[0], title='Flatfield '+channel)
    #plot_CCDimage(flatfield2, fig=fig, axis=ax[1], title='Flatfield2 '+channel)
    
    plot_CCDimage(flatfield_err, fig=fig, axis=ax[1], title='Flatfield error '+channel)

    # if channel in ['IR1','IR3','UV1','UV2']:
    #     plot_CCDimage(flatfield_err-np.fliplr(mean_flatfield_errors_scaled), fig=fig, axis=ax[2], title='Flatfield error minus mean '+channel)

    # else:
    #     plot_CCDimage(flatfield_err-mean_flatfield_errors_scaled, fig=fig, axis=ax[2], title='Flatfield error minus mean '+channel)    

    fig.savefig("output/flatfield_" + channel + ".jpg")



#%%
#%%
    
#The code below was to check if the reported error could be reduced by removing the average of the 
#reported errors with the reasoning htat this was due to the change of lighting on hte screen. 
# This actually made the error worse for the IRchannels though and only improved the UV channels, why it was not used.
flatfields_errors_scaled = {}
for channel in channels:
    if channel in ['IR1','IR3','UV1','UV2']:
        flatfields_errors_scaled[channel]=np.fliplr((flatfields_errors[channel]))
    else:
        flatfields_errors_scaled[channel]=(flatfields_errors[channel])


mean_flatfield_errors_scaled=(flatfields_errors_scaled['IR1']+flatfields_errors_scaled['IR2']+flatfields_errors_scaled['IR3']+flatfields_errors_scaled['IR4']+flatfields_errors_scaled['UV1']+flatfields_errors_scaled['UV2'])/6

   
plot_CCDimage(mean_flatfield_errors_scaled, title='Mean flatfield error')
plt.plot(mean_flatfield_errors_scaled.mean(0))





# %%
