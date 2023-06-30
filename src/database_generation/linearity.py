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
from database_generation.experimental_utils import filter_on_time
import pickle
import mats_l1_processing.instrument as instrument
from mats_l1_processing.read_in_functions import channel_num_to_str, add_and_rename_CCDitem_info
from mats_l1_processing.L1_calibration_functions import CCD, inverse_model_real,total_model_scalar,check_true_value_max,test_for_saturation

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

    return p,params[1]


def fit_with_spline(x, y, deg):
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(6)
    return my_pwlf


def fit_curve(man_tot, inst_tot, threshold=np.inf, fittype="polyfit1",inverse=False):
    """Bins the data into evenly spaced bins and performs a fit. Valid fittypes are:
    'polyfit1'
    'polyfit2'
    'threshold2'

    2022.08.22: Only threshold 2 is valid (OMC)
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
    elif fittype == "threshold2":
        p_low,covariance = fit_with_curvefit(
            x,
            y,
            2,
            'threshold',
        )    
    else:
        ValueError("Invalid fittype")
    
    return p_low, bin_center, low_measured_mean,covariance

def get_linearity(
    CCDitems,
    calibration_file,
    channels=[1, 2, 3, 4, 5, 6],
    testtype="col",
    fittype="threshold2",
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
        
        poly_or_spline, bin_center, low_measured_mean,covariance = fit_curve(
            man_tot, inst_tot, threshold, fittype
        )
        
        #Generate non-linearity object and save it
        non_linearity = instrument.nonLinearity(channels[i],fittype=fittype, 
                        fit_parameters=poly_or_spline, covariance=covariance, fit_threshold=threshold,
                        dfdx_non_lin_important=0.4,dfdx_saturation=0.05)

        filename = 'linearity' + '_' + testtype + '_' + str(channels[i]) + '.pkl'    
        with open(filename, 'wb') as f:
            pickle.dump(non_linearity, f)
        
        #Plotting
        if plot:
            plt.plot([0, threshold*1.3], [0, threshold*1.3], "k--",label=None)
            
            plt.xlabel('simulated (counts)')
            plt.ylabel('measured (counts)')

            plt.plot(
                man_tot.flatten()[::plotting_factor],
                inst_tot[::plotting_factor],
                ".",
                alpha=0.1,
                markeredgecolor="none",
                c=color[channels[i]],
                label=None,
            )
            plt.plot(
                bin_center,
                low_measured_mean,
                "+",
                c=color[channels[i]],
                label=None,
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
                    label='channel ' + str(channels[i]),
                )

            else:
                plt.plot(
                    np.arange(0, threshold*1.3),
                    np.polyval(poly_or_spline, np.arange(0, threshold*1.3)),
                    "-",
                    c=color[channels[i]],
                    label='channel' + str(channels[i]),
                )

            plt.plot([0, threshold], [threshold, threshold], "k:",label=None)
            plt.xlim([0, threshold*1.3])
            plt.ylim([0, threshold*1.3])

    if plot:
        plt.legend()
        plt.grid(True)
        plt.savefig("linearity_fit_" + testtype + ".png")
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
        plot=plot
    )

    return poly_or_spline

def gen_non_linear_table(CCDitem,calibrationfile=None,fittype='interp',randomize=False):
    """ 
    Generates a table of "real" counts (corrected for non-linearity) for measured 
    values from 0-2^16-1 for a given CCDitem taking into account binning to combine 
    the different non-linearity coefficients from pixel, row and column.

    Author: Ole Martin Christensen
    
    Args:
        CCDitem (dict): CCDitem
        calibrationfile (str):  calibration file which specifies where to get the non-lin
                                constants from.
        fittype (str):  whether to simply tabulate the values ('interp') or
                        use an inverse model ('inverse'). Default: 'interp'.
        randomize (bool): whether to perturb the non-lin constants with their covariance 
                        before caluculating the table
    Returns: 
        
        x_true (np.array, dtype=float64): The true counts after non-lin correction
        x_measured (np.array, dtype=int): Measured counts
        CCDitem (dict): Copy of the CCDItem
        flag (np.array, dtype=int): Flag to mark saturation (0,1 or 3). See test_for_saturation

    """


    #Add stuff (if not allready there) required for add_and_rename.
    if not ("read_from" in CCDitem):
        CCDitem["read_from"] = 'rac'
    if not ("EXP Nanoseconds" in CCDitem):
        CCDitem["EXP Nanoseconds"] = 0
    if not ("BC") in CCDitem:
        CCDitem["BC"] = '[]'

    CCDitem = add_and_rename_CCDitem_info(CCDitem)
    if calibrationfile==None:
        pass
    else:
        CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)
    
    CCDitem["CCDunit"].non_linearity_pixel.non_lin_important = CCDitem["CCDunit"].non_linearity_pixel.calc_non_lin_important(0.5)
    CCDitem["CCDunit"].non_linearity_sumrow.non_lin_important = CCDitem["CCDunit"].non_linearity_sumrow.calc_non_lin_important(0.5)
    CCDitem["CCDunit"].non_linearity_sumwell.non_lin_important = CCDitem["CCDunit"].non_linearity_sumwell.calc_non_lin_important(0.5)


    if randomize:
        CCDitem["CCDunit"].non_linearity_pixel.fit_parameters = CCDitem["CCDunit"].non_linearity_pixel.get_random_fit_parameter()
        CCDitem["CCDunit"].non_linearity_sumwell.fit_parameters = CCDitem["CCDunit"].non_linearity_sumwell.get_random_fit_parameter()

        CCDitem["CCDunit"].non_linearity_pixel.saturation = CCDitem["CCDunit"].non_linearity_pixel.calc_non_lin_important(0.05)
        CCDitem["CCDunit"].non_linearity_sumrow.saturation = CCDitem["CCDunit"].non_linearity_sumrow.calc_non_lin_important(0.05)
        CCDitem["CCDunit"].non_linearity_sumwell.saturation = CCDitem["CCDunit"].non_linearity_sumwell.calc_non_lin_important(0.05)

    x_true,x_measured,CCDitem,flag = tabulate_non_linearity(CCDitem,fittype)

    return x_true,x_measured,CCDitem,flag

def tabulate_non_linearity(CCDitem,fittype='interp'):
    try:
        CCDunit = CCDitem["CCDunit"]
    except:
        raise Exception("No CCDunit defined for the CCDitem")

    nrowbin = CCDitem["NRBIN"]
    ncolbin = CCDitem["NCBIN CCDColumns"]

    if fittype=='interp':
        x_true_samples = np.arange(0,2**17-1,1)
        x_measured_sample = np.zeros(x_true_samples.shape)

        for i in range(len(x_true_samples)):
            x_measured_sample[i] = total_model_scalar(x_true_samples[i],CCDunit,nrowbin,ncolbin)

        x_measured = np.arange(0,2**16-1,1)
        x_true = np.interp(x_measured,x_measured_sample,x_true_samples)
        flag = np.zeros(x_true.shape)

        for i in range(len(x_true)):
            flag[i], x_sat = test_for_saturation(CCDunit,nrowbin,ncolbin,x_measured[i])
            if (flag[i] == 3):
                x_true[i] = x_sat

            flag[i],x_true[i] = check_true_value_max(CCDunit,nrowbin,ncolbin,x_true[i],flag[i])

    elif fittype=='inverse':
        x_measured = np.arange(0,2**16-1,1)
        x_true = np.zeros(x_measured.shape)
        flag = np.zeros(x_measured.shape)
        
        for i in range(len(x_measured)):
            x_true[i],flag[i] = inverse_model_real(CCDitem,x_measured[i],method='Nelder-Mead')

    else:
        raise ValueError('fittype need to be inverse or interp')

    return x_true,x_measured,CCDitem,flag

def add_table(CCDitem,fittype='interp'):
    x_true,x_measured,CCDitem, flag = tabulate_non_linearity(CCDitem,fittype)
    table = np.array([x_true, flag,x_measured])
    tablefilename = str("channel_" +  str(CCDitem["CCDSEL"]) + "_nrbin_"
                    + str(CCDitem["NRBIN"]) + "_ncbinfpga_" + str(CCDitem["NCBIN FPGAColumns"]) 
                    + "_ncbinccd_" +str(CCDitem["NCBIN CCDColumns"]))
    
    np.save(CCDitem["CCDunit"].tablefolder + tablefilename + '.npy',table)
    
    df = pd.read_csv(CCDitem["CCDunit"].tablefolder + 'tables.csv')
    
    df.loc[len(df.index)] = [CCDitem["CCDSEL"],       
    CCDitem["WDW Mode"],
    CCDitem["WDW InputDataWindow"],
    CCDitem["WDWOV"],
    CCDitem["JPEGQ"],
    CCDitem["FRAME"],
    CCDitem["NROW"],
    CCDitem["NRBIN"],
    CCDitem["NRSKIP"],
    CCDitem["NCOL"],
    CCDitem["NCBIN FPGAColumns"],
    CCDitem["NCBIN CCDColumns"],
    CCDitem["NCSKIP"],
    CCDitem["NFLUSH"],
    CCDitem["TEXPMS"],
    CCDitem["GAIN Mode"],
    CCDitem["GAIN Timing"],
    CCDitem["GAIN Truncation"],
    CCDitem["TEMP"],
    CCDitem["FBINOV"],
    CCDitem["LBLNK"],
    CCDitem["TBLNK"],
    CCDitem["ZERO"],
    CCDitem["TIMING1"],
    CCDitem["TIMING2"],
    CCDitem["VERSION"],
    CCDitem["TIMING3"],
    tablefilename]

    df.to_csv(CCDitem["CCDunit"].tablefolder + 'tables.csv',index=False)

    return