#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 07:28:46 2020

Forward model for MATS' calibration

@author: lindamegner
"""

import numpy as np
import matplotlib.pyplot as plt

from database_generation.experimental_utils import readprotocol 
from database_generation.experimental_utils import read_all_files_in_protocol, get_true_image_reverse, desmear_true_image_reverse
from mats_l1_processing.L1_calibration_functions import get_true_image, desmear_true_image, subtract_dark, flatfield_calibration
from mats_l1_processing.L1_calibration_functions import calculate_flatfield, calculate_dark, bin_image_with_BC
from mats_l1_processing.instrument import Instrument, CCD
#from experimental_utils import plot_CCDimage 

def frameplot(pic,fig,axis,title='',clim=999):

                    
    sp=axis.pcolormesh(pic,cmap=plt.cm.jet)
    axis.set_title(title)
    if clim==999:
        mean=pic.mean()
        std=pic.std()
        sp.set_clim([mean-1*std,mean+1*std])
       
    else:
        sp.set_clim(clim)

    fig.colorbar(sp,ax=axis)

    return sp    
    


def forward(photons,CCDitem, f=0, plotme=True):
    simage_raw=np.float64(photons*np.ones([511,2048]))
    #simage_raw=np.float64(photons*np.ones_like(CCDitem['IMAGE']))
    # TOD0 Step 8 Transform from photons to electrons and then to LSB.   

    #if CCDitem['NCBIN CCDColumns']>1 or CCDitem['NCBIN FPGAColumns']>1 or CCDitem['NRBIN']>1 : 
    simage_raw_binned=bin_image_with_BC(CCDitem.copy(), simage_raw.copy())
   # simage_raw_binned=get_true_image(CCDitem.copy(),simage_raw_binned)  
    
    #plotmean=photons*CCDitem['NCBIN CCDColumns']*CCDitem['NCBIN FPGAColumns']*CCDitem['NRBIN']
    #clims=[plotmean-np.sqrt(plotmean), plotmean+np.sqrt(plotmean)]
    
    # Now modify the image in forward direction and Plot the result
    
    
    # Step 7 Add ghost imaging. TBD.
    
    
    #flatfield
    image_flatf_fact=calculate_flatfield(CCDitem.copy())     

    simage_flatf_binned=simage_raw_binned*image_flatf_fact*CCDitem["CCDunit"].calib_denominator(CCDitem["GAIN Mode"])

    #dark
    # TBD: Decide on threshold fro when to use pixel correction (little dark current) and when to use average image correction (large dark current). 
    # TBD: The temperature needs to be decided in a better way then taken from the ADC as below.
    # Either read from rac files of temperature sensors or estimated from the top of the image
    
    simage_dark_binned=simage_flatf_binned+calculate_dark(CCDitem.copy())
    #dark_fullpic=calculate_dark(CCDitem.copy())    
    #simage_dark=simage_flatf+dark_fullpic[CCDitem['NRSKIP']:CCDitem['NRSKIP']+CCDitem['NROW'],
    #                                      CCDitem['NCSKIP']:CCDitem['NCSKIP']+CCDitem['NCOL']+1]


    
    #add smear
    simage_smear=desmear_true_image_reverse(CCDitem.copy(), simage_dark_binned.copy())

        
    #add bias
    simage_bias=get_true_image_reverse(CCDitem.copy(),simage_smear.copy())

    
    return simage_raw,simage_raw_binned,  simage_flatf_binned, simage_dark_binned, simage_smear, simage_bias
        
       
    

def backward(input_image,CCDitem, b=1, d=2, plotme=True):    
    
        # # Do normal calibration to reverse the forward model   
    image=input_image.copy()

    image_bias_sub, flags = get_true_image(CCDitem, image)

    image_desmeared, flags = desmear_true_image(CCDitem,image_bias_sub.copy())

    image_dark_sub, flags=subtract_dark(CCDitem,image_desmeared.copy())
        
    image_calib_nonflipped, flags=flatfield_calibration(CCDitem,image_dark_sub.copy())
    #plotmean=photons*CCDitem['NCBIN CCDColumns']*CCDitem['NCBIN FPGAColumns']*CCDitem['NRBIN']
    #clims=[plotmean-np.sqrt(plotmean), plotmean+np.sqrt(plotmean)]


    
   
    

    return image, image_bias_sub, image_desmeared, image_dark_sub, image_calib_nonflipped

def forward_and_backward(CCDitem, photons, plot=True):
    #clims=[-2,2]



    
    simage_raw,simage_raw_binned, simage_flatf_binned, simage_dark_binned, simage_smear, simage_bias=forward(photons,CCDitem, plotme=False)

    if plot:
        fig,ax=plt.subplots(5,3)
        f=0
        frameplot(simage_raw_binned, fig, ax[0,f], title='raw simulated image')
        frameplot(simage_flatf_binned, fig, ax[1,f], title='raw+flat')
    
        frameplot(simage_dark_binned, fig, ax[2,f], title='raw+flat+dark+binned')
    
        frameplot(simage_smear, fig, ax[3,f], title='raw+flat+dark+smear')
        frameplot(simage_bias,fig, ax[4,f], title='raw+flat+dark+smear+bias')



    image, image_bias_sub, image_desmeared, image_dark_sub, image_calib_nonflipped=backward(simage_bias,CCDitem, plotme=False)

    if plot:
        b=1
        d=2
        frameplot(image,fig, ax[4,b], 'From forward')
        frameplot(image-image,fig, ax[4,d], 'simage-image')
    
        frameplot(image_bias_sub,fig, ax[3,b], 'Bias subtracted') 
        frameplot(simage_smear-image_bias_sub,fig, ax[3,d], 'simage-image')
    
        frameplot(image_desmeared,fig, ax[2,b],' Desmeared LSB')  
        frameplot(simage_dark_binned-image_desmeared,fig, ax[2,d], 'simage-image')
    
        frameplot(image_dark_sub,fig, ax[1,b], ' Dark current subtracted.')     
        frameplot(simage_flatf_binned-image_dark_sub,fig, ax[1,d], 'simage-image')
    
        frameplot(image_calib_nonflipped,fig, ax[0,b], ' Flat field compensated.')     
        frameplot(simage_raw_binned-image_calib_nonflipped,fig, ax[0,d], 'simage-image')
    
        fig.suptitle('Forward model followed by backward i.e calibration')

    #Test to see that the backward removes what forward added:
    assert np.sum(np.abs(simage_dark_binned-image_desmeared)) < image.size*1.e-12  
    assert np.sum(np.abs(simage_flatf_binned-image_dark_sub)) < image.size*1.e-12  
    assert np.sum(np.abs(simage_raw_binned-image_calib_nonflipped)) < image.size*1.e-12  
                   
# =============================================================================
# Main
# =============================================================================

"""




# Read in a CCDitem 



#directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Flatfields/Diffusor/DiffusorFlatTests/'
#protocol='ForwardModelTestProto.txt'


directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Binning/Binning-simulation/'
protocol='PROTOCOL-BINNING.txt'
calibration_file='/Users/lindamegner/MATS/retrieval/git/MATS-L1-processing/scripts/calibration_data_linda.toml'


read_from='rac'  
df_protocol=readprotocol(directory+protocol)
df_bright=df_protocol[df_protocol.DarkBright=='B']
CCDitems=read_all_files_in_protocol(df_bright, read_from,directory)
# The image of this CCDitem is not used  , only the meta data


CCDitem=CCDitems[0]


#  Hack to have no compensation for bad colums at the time. This should nolonger be needed LM 28Jul2022
# CCDitem['NBC']=0
# CCDitem['BC']=np.array(CCDitem['BC'])  
    

intrument = Instrument(calibration_file)
CCDitem['CCDunit']=intrument.get_CCD("IR1")

forward_and_backward(CCDitem,  photons=1000, plot=True)

"""