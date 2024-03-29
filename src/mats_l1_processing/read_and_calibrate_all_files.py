#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:33:31 2020

@author: Linda Megner, Ole Martin Christensen

Main function, loops though all images in a folder an calibrates them

"""

# %%

from database_generation.experimental_utils import plotCCDitem


from mats_l1_processing.read_in_functions import read_CCDitems
import matplotlib.pyplot as plt

from mats_l1_processing.L1_calibrate import L1_calibrate
from database_generation.experimental_utils import plot_CCDimage
from mats_l1_processing.instrument import Instrument

import argparse
from tqdm import tqdm


def main(directory, calibration_file, calibrate=True, plot=False):
    """Run program.

    Keyword arguments:
    directory -- input directory
    """

    
    instrument = Instrument(calibration_file)

    
    CCDitems = read_CCDitems(directory)  # read in data

    # calibrate and/or plot the images
    if calibrate:
        for CCDitem in tqdm(CCDitems):
            (
                image_lsb,
                image_bias_sub,
                image_desmeared,
                image_dark_sub,
                image_calib_nonflipped,
                image_calibrated,
                image_common_fov,
                errors,
            ) = L1_calibrate(CCDitem, instrument)

            if plot:
                fig, ax = plt.subplots(5, 1)
                plot_CCDimage(image_lsb, fig, ax[0], "Original LSB")
                plot_CCDimage(image_bias_sub, fig, ax[1], "Bias subtracted")
                plot_CCDimage(image_desmeared, fig, ax[2], " Desmeared LSB")
                plot_CCDimage(
                    image_dark_sub, fig, ax[3], " Dark current subtracted LSB"
                )
                plot_CCDimage(
                    image_calib_nonflipped, fig, ax[4], " Flat field compensated LSB"
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
    
    parser.add_argument(
        "-calib_file",
        nargs="?",
        default="../../scripts/calibration_data.toml",
        dest="calib_file",
        help="calibration toml file",
    )
    parser.add_argument(
        "-nc",
        "-no_calibrate",
        action="store_false",
        dest="no_calibrate",
        help="flag to calibrate",
    )
    parser.add_argument(
        "-p",
        "-plot",
        action="store_true",
        dest="plot",
        help="flag to plot",
    )

    args = parser.parse_args()
    main(args.in_directory, args.no_calibrate, args.plot)

# %%
