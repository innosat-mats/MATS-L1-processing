"""
Created on Thu Oct 07 09:47 2021

@author: olemartinchristensen

Functions are used to estimate the linearity of the MATS channels from binning tests
"""
from database_generation import binning_functions as bf
import pandas as pd
from matplotlib import pyplot as plt
from mats_l1_processing import read_in_functions
import numpy as np
from scipy.stats import binned_statistic
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import toml
from mats_l1_processing.LindasCalibrationFunctions import filter_on_time


def fit_with_polyfit(x, y, deg):

    p = np.polyfit(
        x,
        y,
        deg,
        full=True,
    )[0]

    return p


# two functions to fit through origin


def linear_fit(x, a):
    # Curve fitting function
    return a * x  # b=0 is implied


def quadratic_fit(x, a, b):
    # Curve fitting function
    return a * x ** 2 + b * x  # c=0 is implied


def fit_with_curvefit(x, y, deg):

    if deg == 1:
        params = curve_fit(linear_fit, x, y)
        [a] = params[0]
        p = [a, 0]
    elif deg == 2:
        params = curve_fit(quadratic_fit, x, y)
        [a, b] = params[0]
        p = [a, b, 0]
    else:
        ValueError("only deg 1 and 2 are accepted")

    return p


def fit_with_spline(x, y, deg):
    spl = UnivariateSpline(x, y, k=1, s=0.01)
    return spl


def fit_curve(man_tot, inst_tot, threshold=np.inf, fittype="polyfit"):
    """Bins the data into evenly spaced bins and performs a fit. Valid fittypes are:
    'polyfit'
    'curvefit1'
    'curvefit2'
    'spline1'
    """

    all_simulated = man_tot.flatten()
    all_measured = inst_tot.flatten()

    # fit linear part
    low_simulated = all_simulated[all_simulated < threshold]
    low_measured = all_measured[all_simulated < threshold]

    low_measured_mean, bin_edges = binned_statistic(
        low_simulated, low_measured, "median", bins=2000
    )[0:2]

    bin_center = (bin_edges[1:] + bin_edges[0:-1]) / 2

    if fittype == "polyfit":
        p_low = fit_with_polyfit(
            bin_center[~np.isnan(low_measured_mean)],
            low_measured_mean[~np.isnan(low_measured_mean)],
            1,
        )
    elif fittype == "curvefit1":
        p_low = fit_with_curvefit(
            bin_center[~np.isnan(low_measured_mean)],
            low_measured_mean[~np.isnan(low_measured_mean)],
            1,
        )
    elif fittype == "curvefit2":
        p_low = fit_with_curvefit(
            bin_center[~np.isnan(low_measured_mean)],
            low_measured_mean[~np.isnan(low_measured_mean)],
            2,
        )
    elif fittype == "spline1":
        p_low = fit_with_spline(
            bin_center[~np.isnan(low_measured_mean)],
            low_measured_mean[~np.isnan(low_measured_mean)],
            1,
        )
    else:
        ValueError("Invalid fittype")

    return p_low, bin_center, low_measured_mean


def get_linearity(
    CCDitems, testtype="col", plot=True, fittype="polyfit", channels=[1, 2, 3, 4, 5, 6]
):
    plotting_factor = 1
    threshold = 30e3

    color = cm.rainbow(np.linspace(0, 1, 7))

    for i in range(len(channels)):
        (
            man_tot,
            inst_tot,
            channel,
            test_type,
        ) = bf.get_binning_test_data_from_CCD_item(
            CCDitems, test_type_filter=testtype, channels=[channels[i]]
        )

        poly_or_spline, bin_center, low_measured_mean = fit_curve(
            man_tot, inst_tot, threshold, fittype
        )

        if plot:
            plt.plot(
                man_tot.flatten()[::plotting_factor],
                inst_tot[::plotting_factor],
                ".",
                alpha=0.1,
                markeredgecolor="none",
                c=color[channels[i]],
            )
            plt.plot(
                bin_center,
                low_measured_mean,
                "+",
                c=color[channels[i]],
            )
            if fittype == "spline1":
                plt.plot(
                    np.arange(0, 40000),
                    poly_or_spline(np.arange(0, 40000)),
                    "-",
                    c=color[channels[i]],
                )
            else:
                plt.plot(
                    np.arange(0, 40000),
                    np.polyval(poly_or_spline, np.arange(0, 40000)),
                    "-",
                    c=color[channels[i]],
                )

        print(poly_or_spline)
    if plot:
        # plt.savefig("linearity_fit_channel_" + str(channels[i]) + ".png")
        plt.show()

    return poly_or_spline


def make_linearity(channel, calibration_file, plot=True):

    calibration_data = toml.load(calibration_file)

    CCDitems = read_in_functions.read_CCDitems(calibration_data["linearity"]["folder"])

    print(len(CCDitems))

    starttime = None
    endtime = None

    if calibration_data["linearity"]["starttime"] != "":
        starttime = pd.to_datetime(
            calibration_data["linearity"]["starttime"], format="%Y-%m-%dT%H:%MZ"
        )

    if calibration_data["linearity"]["endtime"] != "":
        endtime = pd.to_datetime(
            calibration_data["linearity"]["endtime"], format="%Y-%m-%dT%H:%MZ"
        )

    if (starttime != None) or (endtime != None):
        CCDitems = filter_on_time(CCDitems, starttime, endtime)

    # Main function
    poly_or_spline = get_linearity(
        CCDitems,
        "col",
        plot,
        calibration_data["linearity"]["fittype"],
        channels=channel,
    )

    return poly_or_spline
