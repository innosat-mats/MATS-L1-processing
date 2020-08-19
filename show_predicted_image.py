#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:53:14 2019

@author: franzk

"""

import numpy as np
from L1_functions import readimgpath, predict_image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

ref_hsm_image, ref_hsm_header = readimgpath(
    "/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/", 0, 0
)
ref_lsm_image, ref_lsm_header = readimgpath(
    "/home/franzk/Documents/MATS/L1_processing/data/2019-02-08 rand6/", 4, 0
)

image_index = 17

image_display_adjustment = 200


image, header = readimgpath = readimgpath(
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

if header["Ending"] == "Wrong Size":
    print("Something wrong with the image")
else:
    prim, prim_header = predict_image(
        ref_hsm_image, ref_hsm_header, ref_lsm_image, ref_lsm_header, header
    )

    pred_mean_img = np.mean(prim)

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(
        prim,
        cmap="viridis",
        vmin=pred_mean_img - image_display_adjustment,
        vmax=pred_mean_img + image_display_adjustment,
    )
    plt.title("Generated from reference")
    plt.xlabel("Pixels")
    plt.ylabel("Pixels")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

print(
    "Image: "
    + str(image_index)
    + ", CCD image mean: "
    + str(mean_img)
    + ", Predicted image mean: "
    + str(pred_mean_img)
    + ", Blank: "
    + str(header["BlankLeadingValue"])
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

