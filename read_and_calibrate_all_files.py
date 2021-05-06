#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:33:31 2020

@author: lindamegner
"""

# %%

from LindasCalibrationFunctions import plotCCDitem


from read_in_functions import read_all_files_in_directory
import matplotlib.pyplot as plt

from L1_calibrate import L1_calibrate
from LindasCalibrationFunctions import plot_CCDimage

import argparse


def main(directory="data/"):

    read_from = "rac"
    CCDitems = read_all_files_in_directory(read_from, directory)

    calibrate = True
    plot = True

    if calibrate:
        for CCDitem in CCDitems[:3]:
            (
                image_lsb,
                image_bias_sub,
                image_desmeared,
                image_dark_sub,
                image_flatf_comp,
            ) = L1_calibrate(CCDitem)

            if plot == True:
                fig, ax = plt.subplots(5, 1)
                plot_CCDimage(image_lsb, fig, ax[0], "Original LSB")
                plot_CCDimage(image_bias_sub, fig, ax[1], "Bias subtracted")
                plot_CCDimage(image_desmeared, fig, ax[2], " Desmeared LSB")
                plot_CCDimage(
                    image_dark_sub, fig, ax[3], " Dark current subtracted LSB"
                )
                plot_CCDimage(
                    image_flatf_comp, fig, ax[4], " Flat field compensated LSB"
                )
                fig.suptitle(CCDitem["channel"])

    else:

        for CCDitem in CCDitems[:]:
            fig = plt.figure()
            ax = fig.gca()
            plotCCDitem(CCDitem, fig, ax, title=CCDitem["channel"])


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calibrates MATS images")
    parser.add_argument(
        "-i",
        "-in_directory",
        nargs="?",
        default="data/",
        dest="in_directory",
        help="input diretory of mats images and metadata",
    )

    args = parser.parse_args()

    main(args.in_directory)

# %%
