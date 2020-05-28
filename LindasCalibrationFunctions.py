#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 06:48:10 2020

@author: lindamegner


Functions for data analysis of during the MATS calibration

"""

from read_in_functions import read_CCDitem_from_imgview
import matplotlib.pyplot as plt



def plot_CCDimage(image,fig,axis,title='',clim=999):
    import numpy as np
    import matplotlib.pyplot as plt

    sp=axis.pcolormesh(image,cmap=plt.cm.jet)
    if clim==999:
        mean=np.mean(image)
        std=np.std(image)
        sp.set_clim([mean-3*std, mean+3*std])
    else:
        plt.clim(clim)        
    fig.colorbar(sp,ax=axis)
    axis.set_title(title)
    return sp



def plot_simple(filename, clim=999):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    
    pic = np.float64(Image.open(filename)) #read image 
    plt.imshow(pic)
    if clim==999:
        mean=pic.mean()
        std=pic.std()
        plt.clim([mean-2*std,mean+2*std])
    else:
        plt.clim(clim)
    plt.colorbar()



def readandsubtractdark(dirname, imageID, dark1ID, dark2ID='999', multiplydark=1):
# Takes directory and PicID of image and dark picture(s) and subtracts them 

    
    image=read_CCDitem_from_imgview(dirname, imageID)
    dark1=read_CCDitem_from_imgview(dirname, dark1ID)
    if dark2ID=='999':
            imagenew=image['IMAGE']-multiplydark*dark1['IMAGE']
    else:
            dark2=read_CCDitem_from_imgview(dirname, dark2ID)
            imagenew=image['IMAGE']-multiplydark*(dark1['IMAGE']+dark2['IMAGE'])/2.
    return imagenew

    



def plotCCDitem(CCDitem,fig,axis,title='',clim=999):

    pic=CCDitem['IMAGE']                
    sp=axis.pcolormesh(pic,cmap=plt.cm.jet)
    axis.set_title(title)
    if clim==999:
        mean=pic.mean()
        std=pic.std()
        sp.set_clim([mean-2*std,mean+2*std])
    else:
        sp.set_clim(clim)

    fig.colorbar(sp,ax=axis)

    return sp

def UniqueValuesInKey(listofdicts,keystr):    
    uniqueValues = list()
    for x in listofdicts:
        if x[keystr] not in uniqueValues :
            uniqueValues.append(x[keystr])
    return uniqueValues


class ItemsUnitCreate:
    import numpy as np

    def __init__(self,df,dirname):
        self.df=df
        df_B=df[df.DarkBright=='B']
        df_D=df[df.DarkBright=='D'] 
        
        self.imageItem=read_CCDitem_from_imgview(dirname,df_B.PicID.iloc[0])  

# =============================================================================
#         # imageitems=[]
#         # for i, dfitem in enumerate(df_B):
#         #     imageitems.append(read_CCDitem_from_imgview(dirname,dfitem))
#         # darkitems=[]    
#         # for i, dfitem in enumerate(df_D):
#         #     darkitems.append(read_CCDitem_from_imgview(dirname,dfitem))       
#         # self.image=meanimage(imageitems)
#         # self.dark=meanimage(darkitems)
# =============================================================================
        if len(df_B)==1:   
            self.image=self.imageItem['IMAGE']
        elif len(df_B)==2:
            self.image1Item=read_CCDitem_from_imgview(dirname,df_B.PicID.iloc[0]) 
            self.image2Item=read_CCDitem_from_imgview(dirname,df_B.PicID.iloc[1])
            self.image=(self.image1Item['IMAGE']+self.image2Item['IMAGE'])/2.
        else:
            raise Exception(str(len(df_B))+' brightimage(s) in dataframe')

        if len(df_D)==1:
            self.darkItem=read_CCDitem_from_imgview(dirname,df_D.PicID.iloc[0])    
            self.dark=self.darkItem['IMAGE']
        elif len(df_D)==2:
            self.dark1Item=read_CCDitem_from_imgview(dirname,df_D.PicID.iloc[0]) 
            self.dark2Item=read_CCDitem_from_imgview(dirname,df_D.PicID.iloc[1])
            self.dark=(self.dark1Item['IMAGE']+self.dark2Item['IMAGE'])/2.
        else:
            raise Exception(str(len(df_D))+' dark pictures in dataframe')
        self.subpic=self.image-self.dark
             
    def plot(self,fig,axis,whichpic=2,title='',clim=999):
        #whichpic 0 is image, whichpic 1 is dark  whichpic 2 is subpic 

        if whichpic==0:
            pic=self.dark
        elif whichpic==1:
            pic=self.image
        elif whichpic==2:
            pic=self.subpic
        else:
            raise Exception('whichpic must be 1 2 or 3')
                        
        sp=axis.pcolormesh(pic,cmap=plt.cm.jet)
        axis.set_title(title)
        if clim==999:
#            sp.set_clim([0,4000])
            mean=pic.mean()
            std=pic.std()
            sp.set_clim([mean-1*std,mean+1*std])
        else:
            sp.set_clim(clim)

        fig.colorbar(sp,ax=axis)

        return sp


