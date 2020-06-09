#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:41:36 2020

@author: lindamegner
"""

import numpy as np
from L1_calibration_functions import get_true_image, desmear_true_image, CCD
#from L1_calibration_functions import get_true_image_old, desmear_true_image_old
#################################################
#       L1 calibration routine                  #
#################################################


def L1_calibrate(CCDitem):

    #  Hack to have no compensation for bad colums at the time. TODO later.
    CCDitem['NBC']=0
    CCDitem['BC']=np.array([])     
    
    #Step0 Compensate for window mode. This should be done first, because it is done in Mikaels software
    # TODO: Cchek if automatic mode and implement it.
    
    #The below should really be doene for old rac files too (extracted prior to May2020)    
    if CCDitem['read_from']=='imgview':
        if (CCDitem['WinMode']) <=4:
            winfactor=2**CCDitem['WinMode']
        elif (CCDitem['WinMode']) ==7:       
            winfactor=1       # Check that this is what you want!  
        else:
            raise Exception('Undefined Window')
        image_lsb=winfactor*CCDitem['IMAGE']     
    else:    
        image_lsb=CCDitem['IMAGE']   
    

    print('mean image_lsb',np.mean(image_lsb))     
    # Step 1 and 2: Remove bias and compensate form bad columns, image still in LSB
    image_bias_sub = get_true_image(image_lsb, CCDitem)
#    image_bias_sub = get_true_image(CCDitem)
    
    # Step 4: Desmear
    image_desmeared = desmear_true_image(image_bias_sub.copy(), CCDitem)
#    image_desmeared = desmear_true_image(CCDitem)
  
    
    
    # Step 5 Remove dark current
    # TBD: Decide on threshold fro when to use pixel correction (little dark current) and when to use average image correction (large dark current). 
    # TBD: The temperature needs to be decided in a better way then taken from the ADC as below.
    # Either read from rac files of temperature sensors or estimated from the top of the image
    
    
    
    CCDunit=CCD(CCDitem['channel']) #TODO: Remember to clear out the non used bits of CCD /Linda
  
    # print('mean desmeared '+CCDitem['channel']+': ', np.mean(image_desmeared))    
    
    totdarkcurrent=CCDunit.darkcurrent2D(CCDitem['temperature'],CCDitem['SigMode'])*CCDitem['TEXPMS']/1000. # tot dark current in electrons
    totbinpix=CCDitem['NColBinCCD']*2**CCDitem['NColBinFPGA'] #Note that the numbers are described in differnt ways see table 69 in Software iCD
    image_dark_sub=image_desmeared-totbinpix*CCDunit.ampcorrection*totdarkcurrent/CCDunit.alpha_avr(CCDitem['SigMode'])

    
    # Step 6 Remove flat field of the particular CCD. TBD.
    
    # Step 7 Remove ghost imaging. TBD. 
    
    # Step 8 Transform from LSB to electrons and then to photons. TBD.

    return image_lsb, image_bias_sub, image_desmeared, image_dark_sub