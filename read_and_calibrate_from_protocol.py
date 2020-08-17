#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:33:31 2020

@author: lindamegner
"""



from LindasCalibrationFunctions import  plotCCDitem


from LindasCalibrationFunctions import plot_CCDimage 
from read_in_functions import readprotocol, read_all_files_in_protocol
import matplotlib.pyplot as plt
from L1_calibrate import L1_calibrate
#from L1_calibration_functions import CCD




directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Flatfields/Diffusor/DiffusorFlatTests/'
protocol='ABOUT.txt'

#protocol='ProtocolList1.txt'

read_from='rac'  
df_protocol=readprotocol(directory+protocol)

df_bright=df_protocol[df_protocol.DarkBright=='B']
CCDitems=read_all_files_in_protocol(df_bright, read_from,directory)


calibrate=True

#CCDunits={}
for CCDitem in CCDitems[0:4]:
    
    if calibrate:
        # #Check  if the CCDunit has been created. It takes time to create it so it should not be created if not needed
        # try: CCDitem['CCDunit']
        # except: 
        #     try:
        #         CCDunits[CCDitem['channel']]
        #     except:  
        #         CCDunits[CCDitem['channel']]=CCD(CCDitem['channel']) 
        #     CCDitem['CCDunit']=CCDunits[CCDitem['channel']]
        
        image_lsb,image_bias_sub,image_desmeared, image_dark_sub, image_flatf_comp =L1_calibrate(CCDitem)

        fig,ax=plt.subplots(5,1)
        plot_CCDimage(image_lsb,fig, ax[0], 'Original LSB')    
        plot_CCDimage(image_bias_sub,fig, ax[1], 'Bias subtracted')  
        plot_CCDimage(image_desmeared,fig, ax[2],' Desmeared LSB')  
        plot_CCDimage(image_dark_sub,fig, ax[3], ' Dark current subtracted LSB')  
        plot_CCDimage(image_flatf_comp,fig, ax[4], ' Flat field compensated LSB')         
        fig.suptitle(CCDitem['channel'])

    else:    
    
        fig=plt.figure()
        ax=fig.gca()
        plotCCDitem(CCDitem,fig, ax)   
      