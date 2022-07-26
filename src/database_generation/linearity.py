"""
Created on Thu Oct 07 09:47 2021

@author: olemartinchristensen

Functions are used to estimate the linearity of the MATS channels from binning tests
"""
from re import A
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
import pickle
import mats_l1_processing.instrument as instrument
from mats_l1_processing.read_in_functions import channel_num_to_str

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

def threshold_fit(x, a, b, c):
    # Curve fitting function y=a*x, x<=c. y=b*(x-c) + a*c x>c
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i]<=c:
            y[i] = a * x[i]
        elif x[i]>c:   
            y[i] = b * (x[i]-c) + a*c
        else:
            raise ValueError
    return y

def threshold_fit_2(x, a, b, e):
    # Curve fitting function y=a*x, x<=e. y=bx**2 + (a-2be)*x +b*e**2 +  x>e
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i]<=e:
            y[i] = a * x[i]
        elif x[i]>e:   
            y[i] = b*(x[i]-e)**2 + a*(x[i]-e)+a*e
        else:
            raise ValueError
    return y

def fit_with_curvefit(x, y, deg,fun='polynomial'):

    if deg == 1 and fun=='polynomial':
        params = curve_fit(linear_fit, x, y)
        [a] = params[0]
        p = [a, 0]
    elif deg == 2 and fun=='polynomial':
        params = curve_fit(quadratic_fit, x, y)
        [a, b] = params[0]
        p = [a, b, 0]
    elif deg == 1 and fun=='threshold':
        params = curve_fit(threshold_fit, x, y,p0=np.array([1,1,max(x)/3*2]))
        [a, b, c] = params[0]
        p = [a, b, c]

    elif deg == 2 and fun=='threshold':
        params = curve_fit(threshold_fit_2, x, y,p0=np.array([1,-0.0001,max(x)/3*2]))
        [a, b, e] = params[0]
        p = [a, b, e]


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
    'threshold1'
    'threshold2'
    """

    all_simulated = man_tot.flatten()
    all_measured = inst_tot.flatten()

    # fit linear part
    low_simulated = all_simulated[all_simulated < threshold]
    low_measured = all_measured[all_simulated < threshold]

    low_measured_mean, bin_edges = binned_statistic(
        low_simulated, low_measured, "median", bins=1000
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
    elif fittype == "threshold1":
        p_low = fit_with_curvefit(
            x,
            y,
            1,
            'threshold',
        )
    elif fittype == "threshold2":
        p_low = fit_with_curvefit(
            x,
            y,
            2,
            'threshold',
        )    
    else:
        ValueError("Invalid fittype")

    return p_low, bin_center, low_measured_mean

def get_linearity(
    CCDitems,
    calibration_file,
    channels=[1, 2, 3, 4, 5, 6],
    testtype="col",
    fittype="polyfit1",
    threshold=30e3,
    remove_blanks=True,
    plot=True,
):
    
    #Some plotting options
    plotting_factor = 1000
    color = cm.rainbow(np.linspace(0, 1, 7))
    
    #if channels are not list, then make into list
    if not isinstance(channels, (list, tuple, np.ndarray)):
        channels = [channels]


    for i in range(len(channels)):

        
        CCDunit=instrument.CCD(channel_num_to_str(channels[i]),calibration_file)
        
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
            CCD=CCDunit,
            remove_blanks=remove_blanks,
        )
        
        poly_or_spline, bin_center, low_measured_mean = fit_curve(
            man_tot, inst_tot, threshold, fittype
        )
        
        #Generate non-linearity object and save it
        non_linearity = instrument.nonLinearity(channels[i],fittype=fittype, 
                        fit_parameters=poly_or_spline, fit_threshold=threshold,
                        dfdx_non_lin_important=0.2,dfdx_saturation=0.05)

        filename = 'linearity' + '_' + testtype + '_' + str(channels[i]) + '.pkl'    
        with open(filename, 'wb') as f:
            pickle.dump(non_linearity, f)
        
        #Plotting
        if plot:
            plt.plot([0, threshold*1.3], [0, threshold*1.3], "k--")
            
            plt.xlabel('simulated')
            plt.ylabel('measured')

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
                raise NotImplementedError('spline no longer supported')

            if fittype == "threshold2":
                    x = np.arange(0, threshold*1.3)
                    y = threshold_fit_2(x,poly_or_spline[0],poly_or_spline[1],poly_or_spline[2])
                    plt.plot(
                    x,
                    y,
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

    if plot:
        plt.savefig("linearity_fit_" + testtype + ".png")
        plt.grid(True)
        plt.show()
        plt.close()

    if len(channel)==1:
        return poly_or_spline
    else:
        return None

def make_linearity(channel, calibration_file, plot=True, exp_type='col',inverse=False):
    
    calibration_data = toml.load(calibration_file)

    CCDitems = read_in_functions.read_CCDitems(calibration_data["primary_data"]["linearity"]["folder"])
    
    starttime = None
    endtime = None

    if calibration_data["primary_data"]["linearity"]["starttime"] != "":
        starttime = pd.to_datetime(
            calibration_data["primary_data"]["linearity"]["starttime"], format="%Y-%m-%dT%H:%MZ"
        )

    if calibration_data["primary_data"]["linearity"]["endtime"] != "":
        endtime = pd.to_datetime(
            calibration_data["primary_data"]["linearity"]["endtime"], format="%Y-%m-%dT%H:%MZ"
        )

    if (starttime != None) or (endtime != None):
        CCDitems = filter_on_time(CCDitems, starttime, endtime)

    if exp_type == 'exp':
        threshold=6e3
    elif exp_type == 'row':
        threshold=20e3
    elif exp_type == 'col':
        threshold=40e3

    # Main function
    poly_or_spline = get_linearity(
        CCDitems,
        calibration_file,
        channels=channel,
        testtype=exp_type,
        fittype=calibration_data["primary_data"]["linearity"]["fittype"],
        threshold=threshold,
        remove_blanks=True,
        plot=True
    )

    return poly_or_spline
