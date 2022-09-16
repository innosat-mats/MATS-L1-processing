#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:33:31 2020

@author: Linda Megner, Ole Martin Christensen

Main function, loops though all images in a folder an calibrates them

"""

# %%

from mats_l1_processing.experimental_utils import plotCCDitem


from mats_l1_processing.read_in_functions import read_all_files_in_directory
import matplotlib.pyplot as plt

from mats_l1_processing.L1_calibrate import L1_calibrate
from mats_l1_processing.experimental_utils import plot_CCDimage
from mats_l1_processing.instrument import Instrument

import argparse
from tqdm import tqdm
from joblib import Parallel, delayed


def main(directory, calibration_file, calibrate=True, plot=False):
    """Run program.

    Keyword arguments:
    directory -- input directory
    """

    read_from = "rac"  # read from extracted rac file
    
    instrument = Instrument(calibration_file)
    
    CCDitems = read_all_files_in_directory(read_from, directory)  # read in data

    # calibrate and/or plot the images
    #if calibrate:
    #    Parallel(n_jobs=8)(delayed(L1_calibrate)(CCDitem,calibrationfile) for CCDitem in CCDitems)

    if calibrate:
        for CCDitem in CCDitems:
            L1_calibrate(CCDitem,instrument)

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
    main(args.in_directory, args.no_calibrate)

# %%
