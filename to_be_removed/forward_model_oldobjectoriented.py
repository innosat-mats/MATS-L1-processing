#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 07:28:46 2020

Forward model for MATS' calibration

@author: lindamegner
"""

import numpy as np
import matplotlib.pyplot as plt

from L1_calibration_functions import CCD


class SimageItem(object):

    def __init__(self, CCDitem, photons): 
        global CCDunits

        try: CCDunits
        except:CCDunits={}
    
    
        #Check  if the CCDunit has been created. It takes time to create it so it should not be created if not needed
        try: CCDitem['CCDunit']
        except: 
            try:
                CCDunits[CCDitem['channel']]
            except:  
                CCDunits[CCDitem['channel']]=CCD(CCDitem['channel']) 
            CCDitem['CCDunit']=CCDunits[CCDitem['channel']]

        
        
        
        
        self.CCDitem=CCDitem
        # for key in CCDitem:
        #     setattr(self, key, CCDitem[key])
        rawsimage=np.float64(photons*np.ones_like(CCDitem['IMAGE']))

        #Step 8 Transform from photons to electrons and then to LSB.
        self.rawsimage=rawsimage
        self.simage=self.rawsimage
        
        #  Hack to have no compensation for bad colums at the time. TODO later.
        self.CCDitem['NBC']=0
        self.CCDitem['BC']=np.array([])  
        
        
        
        

    # Step 7 Add ghost imaging. TBD.
    # Step 6 Add flat field of the particular CCD. TBD.

    # Step 5 Add dark current
    # TBD: Decide on threshold fro when to use pixel correction (little dark current) and when to use average image correction (large dark current). 
    # TBD: The temperature needs to be decided in a better way then taken from the ADC as below.
    # Either read from rac files of temperature sensors or estimated from the top of the image


    def add_flatf(self):
        from L1_calibration_functions import calculate_flatfield
        image_flatf_fact=calculate_flatfield(self.CCDitem)      
        self.simage=self.simage*image_flatf_fact 

 


    def add_dark(self):
        from L1_calibration_functions import calculate_dark
        self.simage=self.simage+calculate_dark(self.CCDitem)



        
    def add_smear(self):   
        # Step 4: Desmear
        from L1_calibration_functions import desmear_true_image_reverse        
        self.simage=desmear_true_image_reverse(self.CCDitem.copy(), self.simage.copy())
        




    def add_bias(self):
        from L1_calibration_functions import get_true_image_reverse
        

        self.simage=get_true_image_reverse(self.CCDitem.copy(),self.simage.copy())
        # Step 1 and 2: Remove bias and compensate form bad columns, image still in LSB
        #image_bias_sub = get_true_image(image_lsb, CCDitem)
        #    image_bias_sub = get_true_image(CCDitem)
    
        
        # #Hack to fix bad columns
        # CCDitem['NBC']=0
        # CCDitem['BC']=np.array([])     
    
    
    def plot(self,fig,axis,whichpic='simage',title='',clim=999):

        if whichpic=='simage':
            pic=self.simage
        elif whichpic=='rawsimage':
            pic=self.rawsimage
        else:
            raise Exception('whichpic must be image or raw')
                        
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
        

# =============================================================================
# Main
# =============================================================================




from read_in_functions import readprotocol, read_all_files_in_protocol
from L1_calibration_functions import get_true_image, desmear_true_image, subtract_dark, compensate_flatfield
import copy
from LindasCalibrationFunctions import plot_CCDimage 


# Read in a CCDitem 



directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Flatfields/Diffusor/DiffusorFlatTests/'
protocol='ForwardModelTestProto.txt'
read_from='rac'  
df_protocol=readprotocol(directory+protocol)
df_bright=df_protocol[df_protocol.DarkBright=='B']
CCDitems=read_all_files_in_protocol(df_bright, read_from,directory)
# The imagge of this CCDitem is not used  , only the meta data
myCCDitem=CCDitems[0]




photons=4000 
mySimageItem = SimageItem(myCCDitem, photons)


# Now modify the image in forward direction and Plot the result
fig,ax=plt.subplots(5,3)
f=0
b=1
d=2


mySimageItem.plot(fig, ax[0,f], whichpic='rawsimage', title='raw simulated image')
myS0=mySimageItem.simage
mySimageItem.add_flatf()

# image_flatf_comp=compensate_flatfield(mySimageItem.CCDitem.copy(),mySimageItem.simage.copy())
# plot_CCDimage(image_flatf_comp,fig, ax[0,b], ' Flat field reversed. Original?')
# plot_CCDimage(myS0-image_flatf_comp,fig, ax[0,d], 'simage-image')
# diff=myS0-image_flatf_comp
# print('maxdiff', 'mindiff', diff.max(), diff.min())

mySimageItem.plot(fig, ax[1,f], whichpic='simage', title='raw+flat')
myS1=mySimageItem.simage.copy()
mySimageItem.add_dark()

#image_dark_sub=subtract_dark(mySimageItem.CCDitem.copy(),mySimageItem.simage.copy())
#plot_CCDimage(image_dark_sub,fig, ax[1,b], ' Dark current subtracted. Original?')
#plot_CCDimage(myS1-image_dark_sub,fig, ax[1,d], 'simage-image')


mySimageItem.plot(fig, ax[2,f], whichpic='simage', title='raw+flat+dark')
myS2=mySimageItem.simage.copy()
mySimageItem.add_smear()
#image_desmeared = desmear_true_image(mySimageItem.CCDitem.copy(),mySimageItem.simage.copy())
#plot_CCDimage(image_desmeared,fig, ax[2,b],' Desmeared LSB') 
mySimageItem.plot(fig, ax[3,f], whichpic='simage', title='raw+flat+dark+smear')
myS3=mySimageItem.simage.copy()
mySimageItem.add_bias()
#image_bias_sub = get_true_image(mySimageItem.CCDitem.copy(),mySimageItem.simage.copy())
#plot_CCDimage(image_bias_sub,fig, ax[3,b], 'Bias subtracted') 
mySimageItem.plot(fig, ax[4,f], whichpic='simage', title='raw+flat+dark+smear+bias')
myS4=mySimageItem.simage.copy()

#plot_CCDimage(mySimageItem.simage.copy(),fig, ax[4,b], 'From forward')



# # Do normal calibration to reverse the forward model
deSimageItem=copy.copy(mySimageItem)

# fig,ax=plt.subplots(4,1)

image=deSimageItem.simage.copy()    
plot_CCDimage(image,fig, ax[4,b], 'From forward')
diff4=myS4-image
plot_CCDimage(myS4-image,fig, ax[4,d], 'simage-image')

image_bias_sub = get_true_image(deSimageItem.CCDitem, image)
plot_CCDimage(image_bias_sub,fig, ax[3,b], 'Bias subtracted') 
diff3=myS3-image_bias_sub
plot_CCDimage(myS3-image_bias_sub,fig, ax[3,d], 'simage-image')

image_desmeared = desmear_true_image(deSimageItem.CCDitem,image_bias_sub.copy())
plot_CCDimage(image_desmeared,fig, ax[2,b],' Desmeared LSB')  
diff2=myS2-image_desmeared
plot_CCDimage(myS2-image_desmeared,fig, ax[2,d], 'simage-image')

image_dark_sub=subtract_dark(deSimageItem.CCDitem,image_desmeared.copy())
plot_CCDimage(image_dark_sub,fig, ax[1,b], ' Dark current subtracted.')     
diff1=myS1-image_dark_sub
plot_CCDimage(myS1-image_dark_sub,fig, ax[1,d], 'simage-image')




image_flatf_comp=compensate_flatfield(deSimageItem.CCDitem,image_dark_sub.copy())
plot_CCDimage(image_flatf_comp,fig, ax[0,b], ' Flat field compensated.')     
diff0=myS0-image_flatf_comp
plot_CCDimage(myS0-image_flatf_comp,fig, ax[0,d], 'simage-image')

fig.suptitle('Calibrate fo reverse forward model')

# fig2=plt.figure()
# ax2=fig2.gca()
# sp=ax2.pcolormesh(testimage)
# fig2.colorbar(sp,ax=ax2)
#sp.set_clim([])

