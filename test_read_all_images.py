#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:00:39 2019

@author: franzk
"""
import numpy as np
from L1_functions import readimgpath, predict_image, compare_image

Nimages = 100

#index of images to skip
imagetoskip = []

#initialising arrays with NaN values to clearly identify skipped images in final array
blanks_l=np.full((Nimages,1), np.nan)
blanks_t=np.full((Nimages,1), np.nan)
zero_l=np.full((Nimages,1), np.nan)
ncols=np.full((Nimages,1), np.nan)
nrows=np.full((Nimages,1), np.nan)

ncolbin=np.full((Nimages,1), np.nan)
nrowbin=np.full((Nimages,1), np.nan)

p_offsets=np.full((Nimages,1), np.nan)
p_scales=np.full((Nimages,1), np.nan)
p_std=np.full((Nimages,1), np.nan)

ref_hsm_image, ref_hsm_header = readimgpath('/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/', 0, 0)
ref_lsm_image, ref_lsm_header = readimgpath('/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/', 4, 0)

for jj in range(0,Nimages):
    if jj in imagetoskip:
        continue
    try:
        image, header = readimgpath('/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/', jj, 0)
    except:
        print('Image cannot be read\n', jj)
    blanks_l[jj]=int(header['BlankLeadingValue'])
    blanks_t[jj]=int(header['BlankTrailingValue'])
    ncols[jj]=int(header['NCol'])
    nrows[jj]=int(header['NRow'])
    
    ncolbin[jj]=int(header['NRowBinCCD'])
    nrowbin[jj]=int(header['NColBinCCD'])
    
    zero_l[jj]=int(header['ZeroLevel'])

        
    if header['Ending']=='Wrong size' or int(header['BlankLeadingValue'])==0:
        continue
    try:
        prim, prheader=predict_image(ref_hsm_image, ref_hsm_header, ref_lsm_image, ref_lsm_header, header)
        
        t_off, t_scl, t_std = compare_image(prim, image, header)
        
        p_offsets[jj]=t_off
        p_scales[jj]=t_scl
        p_std[jj]=t_std
        
    except Exception:
        print('Image prediction did not work for image', jj)