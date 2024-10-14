"""
Created on Thu Oct 07 09:47 2021

@author: olemartinchristensen

Functions are used to estimate the linearity of the MATS channels from binning tests
"""
from database_generation import binning_functions as bf
import pandas as pd
from mats_l1_processing import read_in_functions
import numpy as np
from scipy.optimize import curve_fit
import toml
from database_generation.experimental_utils import filter_on_time
import mats_l1_processing.instrument as instrument
from mats_l1_processing.read_in_functions import channel_num_to_str


def threshold_fit(x, b, non_lin_point, sat):
    """
    Applies a piecewise threshold fitting model to the input data.

    Parameters:
    - x (array): Input data points.
    - b (float): Scaling factor for the quadratic part of the function.
    - non_lin_point (float): Point where non-linearity begins.
    - sat (float): Saturation point above which the output is constant.


    Returns:
    - y (array): Output data after applying the threshold model.
    """

    y = b*(x-non_lin_point)**2 + x
    y[np.where(x < non_lin_point)] = x[np.where(x < non_lin_point)]
    y[np.where(x > sat)] = b*(sat-non_lin_point)**2 + sat

    return y

def fit_with_curvefit(x, y):

    params = curve_fit(threshold_fit, x, y,p0=np.array([-0.0001,10000,30000]))
    [b,e,sat] = params[0] #fitted parameters
    # sat is actually no used, a fixed value of 32000 (measued counts) is used instead
    p = [b, e, sat]

    return p,params[1]

def point_non_lin_important(b, non_lin_point):
    """
    Calculates a critical point related to the non-linearity in the threshold fitting model.

    Parameters:
    - b (float): Scaling factor for the quadratic part of the function.
    - non_lin_point (float): Point where non-linearity begins.

    Returns:
    - beta (float): Calculated critical point where significant non-linearity effects are observed.
    """
    max_non_linearity = 0.95
    beta = (-np.sqrt(1-max_non_linearity)*np.sqrt(1- (max_non_linearity + 4*b*non_lin_point)) + max_non_linearity + 2*b*non_lin_point - 1)/(2*b)
    return beta

def generate_non_linearity(
    calibration_file,
    channels=[1, 2, 3, 4, 5, 6, 7],
    threshold = 200e3,
    add_bias=True,
    remove_blanks=False,
):
    """
    Processes calibration data to analyze and extract non-linearity characteristics for specified channels.

    Parameters:
    - calibration_file (str): Path to the calibration file in TOML format containing the calibration settings and metadata.
    - channels (list of int): List of channel numbers to analyze. Defaults to [1, 2, 3, 4, 5, 6].
    - threshold (float): Threshold level for processing. Defaults to 200,000.
    - add_bias (bool): Flag to determine whether to add bias to the measurements. Defaults to True.
    - remove_blanks (bool): Flag to indicate whether blank readings should be removed from the data. Defaults to False.

    Returns:
    - non_linearity_data (DataFrame): DataFrame containing non-linearity data for each channel including parameters like 'b', 'e', 'non_lin_important', 'sumwell_saturation', and 'pixel_saturation'.
    - covariance (array): Covariance matrix of the fit parameters obtained from curve fitting.

    The function loads calibration data, filters it based on specified time frames, and processes it through a series of tests to determine the non-linearity characteristics of each channel. It handles different test types ('exp', 'row', 'col') for extracting raw data and computes non-linearity using curve fitting techniques. The results are adjusted for UV and IR channels specifically.
    """
    
    calibration_data = toml.load(calibration_file)
    CCDitems = read_in_functions.read_CCDitems(calibration_data["primary_data"]["linearity"]["folder"])

    #filter away data not to use from the dataset
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


    #%% Put all raw data into a dataframe with test-type and channel
    df = pd.DataFrame()   
    for channel in channels:

        #%% Get non-linear test data
        test_types = ['exp','row','col']
        for index,test_type in enumerate(test_types):
            (
                man_tot,
                inst_tot,
                channel_tot,
                test_type,
                signal_factor
            ) = bf.get_binning_test_data_from_CCD_item(
                CCDitems,
                test_type_filter=test_type,
                channels=[channel],
                add_bias=add_bias,
                remove_blanks=remove_blanks,
            )

            new_df = pd.DataFrame({
            'manual': man_tot,
            'measured': inst_tot,
            'channel': channel_tot,
            'test': test_type,
            'signal_factor': signal_factor
            })

            df = pd.concat([df, new_df], ignore_index=True)

    #%% Generate non-linearity for UV and IR channels seperatelly

    test_types_non_linearity = ['row','col'] #test types to use for non-linearity estimation
    non_linearity_data = pd.DataFrame(columns=["channel", "b", "e", "non_lin_important", "sumwell_saturation","sumrow_saturation","pixel_saturation"]) #dataframe to hold non-linearity data for each channel


    #Theoretical values for saturation
    pixel_saturation = 150e3 / 34 # pixel saturate at 150 electrons
    sumrow_saturation = pixel_saturation*4 #not used
    sumwell_saturation = 32000 #Hard coded to reflect ADC-saturation

    data = pd.DataFrame()
    for i,test_type in enumerate(test_types_non_linearity):
        data = pd.concat([data, df[(df['channel'].isin(channels)) & (df['test'] == test_type)]])
    x = data['manual'].to_numpy()
    y = data['measured'].to_numpy()

    parameters,covariance = fit_with_curvefit(x, y)
    [b,e,sat] = parameters

    non_lin_important = point_non_lin_important(b,e)
    for channel in channels:
        if channel in [5,6]:
            pixel_saturation = 150e3 / 34 * 3/2
        else:
            pixel_saturation =  150e3 / 34 
        new_row = {'channel': channel, 'b': b, 'e': e, 'non_lin_important': non_lin_important, 'sumwell_saturation': sumwell_saturation, 'sumrow_saturation': sumrow_saturation, 'pixel_saturation': pixel_saturation }
        non_linearity_data.loc[len(non_linearity_data)] = new_row

    return non_linearity_data, covariance

def make_linearity(calibration_file):

    non_linearity_data,_ = generate_non_linearity(calibration_file)
    #save as csv
    non_linearity_data.to_csv('linearity.csv')

    return non_linearity_data