#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:34:19 2019

@author: franzk
"""

import numpy as np
import matplotlib.pyplot as plt
from L1_functions import readimgpath, predict_image, get_true_image, desmear_true_image, compensate_bad_columns, get_true_image_from_compensated


# MATS data chain for raw images
# this script deals with:
# 1. Image prediction from reference image according to header
# 2. Applying bad column correction and removing offsets to get "true image"
# 3. Compensating bad columns in MATS payload OBC

image_display_adjustment = 100

# Import two reference images for the CCD in high and low signal modes
# Reference image depends on the temperature

ref_hsm_image, hsm_header = readimgpath('/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/', 0, 0)
ref_lsm_image, lsm_header = readimgpath('/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/', 4, 0)

recv_image, recv_header = readimgpath('/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/', 1, 0)

#Step 1
# Predict the received image from reference image according to header
pred_image, pred_header = predict_image(ref_hsm_image, hsm_header, ref_lsm_image, lsm_header, recv_header)

#Step 2
# Make actual image out both received and predicted images
# These can be compared if no compression is used.

recv_true_image = get_true_image(recv_image, recv_header)
recv_true_image = desmear_true_image(recv_true_image, recv_header)

pred_true_image = get_true_image(pred_image, pred_header)

#Step 3
# Bad column compensation for MATS payload OBC
recv_comp_image = compensate_bad_columns(recv_image, recv_header)

# Step 4
# Getting "true image" from compensated one in MATS payload OBC
true_comp_image = get_true_image_from_compensated(recv_comp_image, recv_header)
true_comp_image = desmear_true_image(true_comp_image, recv_header)

mean_img = np.mean(np.mean(recv_image))

plt.imshow(recv_image, cmap='viridis', vmin=mean_img-image_display_adjustment, vmax=mean_img+image_display_adjustment,  extent=[0,75,0,511], aspect='auto')
plt.title('CCD image')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()

plt.imshow(pred_image, cmap='viridis', vmin=mean_img-image_display_adjustment, vmax=mean_img+image_display_adjustment,  extent=[0,75,0,511], aspect='auto')
plt.title('predicted image')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()

true_mean_img = np.mean(np.mean(recv_true_image))

plt.imshow(recv_true_image, cmap='viridis', vmin=true_mean_img-image_display_adjustment, vmax=true_mean_img+image_display_adjustment, extent=[0,75,0,511], aspect='auto')
plt.title('CCD true image')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()

plt.imshow(pred_true_image, cmap='viridis', vmin=true_mean_img-image_display_adjustment, vmax=true_mean_img+image_display_adjustment, extent=[0,75,0,511], aspect='auto')
plt.title('predicted true image')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()

plt.imshow(recv_comp_image, cmap='viridis', vmin=mean_img-image_display_adjustment, vmax=mean_img+image_display_adjustment, extent=[0,75,0,511], aspect='auto')
plt.title('compensated in OBC image')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()

plt.imshow(true_comp_image, cmap='viridis', vmin=true_mean_img-image_display_adjustment, vmax=true_mean_img+image_display_adjustment, extent=[0,75,0,511], aspect='auto')
plt.title('True image after compensation in OBC software')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()
