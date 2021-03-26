#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:43:11 2020

@author: lindamegner
"""


import pandas as pd 


#import cv2
import glob
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from LindasCalibrationFunctions import plot_CCDimage

directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Flatfields/20200511_temperature_dependence/RacFiles_out/'
#MyArchive1_20200511-180116_20200511-181223_276595658462524.json
#MyArchive1_20200511-180116_20200511-181223_27659565462524.png
#directory='/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Flatfields/Diffusor/DiffusorFlatTests/RacFilesOut/'

CCDfile='CCD.csv'
HTRfile='HTR.csv'
CPRUfile='CPRU.csv'


#df = pd.read_csv(directory+CCDfile, skiprows=[0]) 
df = pd.read_csv(directory+HTRfile, skiprows=[0]) 

df.head()


list_of_dicts=df.to_dict('records')

# # User list comprehension to create a list of lists from Dataframe rows
# list_of_rows = [dict(row) for row in df.values]
 
# # Print list of lists i.e. rows
# print(list_of_rows)




#image = np.load(directory+str(item['imagefile']) + '_data.npy')



maxplot=5
for image_path in glob.glob(directory+"*.png")[:maxplot]:
    image = mpimg.imread(image_path)
    print(image.shape)
    print(image.dtype)
    fig=plt.figure()
    plot_CCDimage(image,fig,fig.gca())