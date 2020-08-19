#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:54:41 2019

@author: franzk
"""

import numpy as np
from L1_functions import readimgpath, compensate_bad_columns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

image_index = 9
image_display_adjustment = 50

image, header = readimgpath(
    "/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/", image_index, 0
)

mean_img = np.mean(image)

plt.figure()
ax = plt.gca()
im = ax.imshow(
    image,
    cmap="viridis",
    vmin=mean_img - image_display_adjustment,
    vmax=mean_img + image_display_adjustment,
)
plt.title("CCD image")
plt.xlabel("Pixels")
plt.ylabel("Pixels")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

prim = compensate_bad_columns(image, header)

pred_mean_img = np.mean(prim)

plt.figure()
ax = plt.gca()
im = ax.imshow(
    prim,
    cmap="viridis",
    vmin=pred_mean_img - image_display_adjustment,
    vmax=pred_mean_img + image_display_adjustment,
)
plt.title("With bad columns compensated")
plt.xlabel("Pixels")
plt.ylabel("Pixels")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

print(
    "Leading blank: "
    + str(header["BlankLeadingValue"])
    + ", Trailing blank: "
    + str(header["BlankTrailingValue"])
)
print(
    "nrowbin: "
    + str(header["NRowBinCCD"])
    + ", ncolbinC: "
    + str(header["NColBinCCD"])
    + ", ncolbinF: "
    + str(header["NColBinFPGA"])
    + ", gain: "
    + str(header["Gain"])
)

