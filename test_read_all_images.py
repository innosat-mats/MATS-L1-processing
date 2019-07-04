#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:00:39 2019

@author: franzk
"""
import numpy as np
import traceback
from L1_functions import readimgpath, predict_image, compare_image

Nimages = 50

imagetoskip = [0]

blanks_l=np.zeros((Nimages,1))
blanks_t=np.zeros((Nimages,1))
zero_l=np.zeros((Nimages,1))
ncols=np.zeros((Nimages,1))
nrows=np.zeros((Nimages,1))

ncolbin=np.zeros((Nimages,1))
nrowbin=np.zeros((Nimages,1))

p_offsets=np.zeros((Nimages,1))
p_scales=np.zeros((Nimages,1))
p_std=np.zeros((Nimages,1))

ref_hsm_image, ref_hsm_header = readimgpath('/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/', 0, 0)
ref_lsm_image, ref_lsm_header = readimgpath('/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/', 4, 0)

for jj in range(0,Nimages):
    #if ismember(jj, imagetoskip):
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
        traceback.print_exc()
        print('Image prediction did not work for image\n', jj)