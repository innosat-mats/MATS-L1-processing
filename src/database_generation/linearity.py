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
from scipy.interpolate import CubicSpline
import scipy.optimize as opt
import pwlf
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
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(6)
    return my_pwlf


def fit_curve(man_tot, inst_tot, threshold=np.inf, fittype="polyfit1",inverse=False):
    """Bins the data into evenly spaced bins and performs a fit. Valid fittypes are:
    'polyfit1'
    'polyfit2'
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

    if not inverse:
        x = bin_center[~np.isnan(low_measured_mean)]
        y = low_measured_mean[~np.isnan(low_measured_mean)]
    elif inverse:
        x = low_measured_mean[~np.isnan(low_measured_mean)]
        y = bin_center[~np.isnan(low_measured_mean)]

    if fittype == "polyfit1":
        p_low = fit_with_polyfit(
            x,
            y,
            1,
        )
    if fittype == "polyfit2":
        p_low = fit_with_polyfit(
            x,
            y,
            2,
        )
    elif fittype == "curvefit1":
        p_low = fit_with_curvefit(
            x,
            y,
            1,
        )
    elif fittype == "curvefit2":
        p_low = fit_with_curvefit(
            x,
            y,
            2,
        )
    elif fittype == "spline1":
        p_low = fit_with_spline(
            x,
            y,
            1,
        )
    else:
        ValueError("Invalid fittype")

    return p_low, bin_center, low_measured_mean


def get_linearity(
    CCDitems,
    testtype="col",
    plot=True,
    fittype="polyfit1",
    channels=[1, 2, 3, 4, 5, 6],
    threshold=30e3,
    remove_blanks=True,
    inverse=False,
):
    plotting_factor = 5

    color = cm.rainbow(np.linspace(0, 1, 7))
    if not isinstance(channels, (list, tuple, np.ndarray)):
        channels = [channels]

    for i in range(len(channels)):
        (
            man_tot,
            inst_tot,
            channel,
            test_type,
        ) = bf.get_binning_test_data_from_CCD_item(
            CCDitems,
            test_type_filter=testtype,
            channels=[channels[i]],
            add_bias=True,
            remove_blanks=remove_blanks,
        )

        poly_or_spline, bin_center, low_measured_mean = fit_curve(
            man_tot, inst_tot, threshold, fittype,inverse
        )

        if plot:
            plt.plot([0, threshold*1.3], [0, threshold*1.3], "k--")
            
            if inverse:
                plt.xlabel('measured')
                plt.xlabel('simulated')

                plt.plot(
                    inst_tot[::plotting_factor],
                    man_tot.flatten()[::plotting_factor],
                    ".",
                    alpha=0.1,
                    markeredgecolor="none",
                    c=color[channels[i]],
                )
                plt.plot(
                    low_measured_mean,
                    bin_center,
                    "+",
                    c=color[channels[i]],
                )
            else:
                plt.xlabel('simulated')
                plt.xlabel('measured')

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
                    np.arange(0, threshold*1.3),
                    poly_or_spline.predict(np.arange(0, threshold*1.3)),
                    "-",
                    c=color[channels[i]],
                )
            else:
                plt.plot(
                    np.arange(0, threshold*1.3),
                    np.polyval(poly_or_spline, np.arange(0, threshold*1.3)),
                    "-",
                    c=color[channels[i]],
                )

            plt.plot([0, threshold], [threshold, threshold], "k:")
            plt.xlim([0, threshold*1.3])
            plt.ylim([0, threshold*1.3])

        np.save(('linearity_' + str(channels[i]) + '_' + testtype),poly_or_spline)

    if plot:
        plt.savefig("linearity_fit_channel_" + str(channels[i]) + ".png")
        plt.grid(True)
        plt.show()

    return poly_or_spline

def make_linearity(channel, calibration_file, plot=True, exp_type='col',inverse=False):

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

    if exp_type == 'exp':
        threshold=4e3
    elif exp_type == 'row':
        threshold=15e3
    elif exp_type == 'col':
        threshold=30e3

    # Main function
    poly_or_spline = get_linearity(
        CCDitems,
        exp_type,
        plot,
        calibration_data["linearity"]["fittype"],
        channels=channel,
        remove_blanks=calibration_data["linearity"]["remove_blanks"],
        inverse=inverse,
        threshold=threshold,
    )

    return poly_or_spline
