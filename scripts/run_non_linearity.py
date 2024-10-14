'''
Plots the data-from the all the non-linearity tests for a single channel
'''
#%%
import numpy as np
from mats_l1_processing.read_in_functions import channel_num_to_str
from database_generation import binning_functions as bf
from database_generation import linearity as linearity
from mats_l1_processing import read_in_functions
import mats_l1_processing.instrument as instrument
import pandas as pd
from database_generation.experimental_utils import filter_on_time
import toml
import os
from scipy.optimize import curve_fit


def threshold_fit(x, b, non_lin_point, sat):
    """
    Applies a piecewise threshold fitting model to the input data.

    Parameters:
    - x (array): Input data points.
    - sat (float): Saturation point above which the output is constant.
    - b (float): Scaling factor for the quadratic part of the function.
    - non_lin_point (float): Point where non-linearity begins.

    Returns:
    - y (array): Output data after applying the threshold model.
    """
    y = b*(x-non_lin_point)**2 + x
    y[np.where(x < non_lin_point)] = x[np.where(x < non_lin_point)]
    y[np.where(x > sat)] = b*(sat-non_lin_point)**2 + sat

    return y

def threshold_fit_inv(y, sat, b, e):
    """
    Inverse of the threshold fitting function, used to retrieve the original data points from the modeled values.

    Parameters:
    - y (array): Modeled data points from which to retrieve original data.
    - sat (float): Saturation point used in the forward model.
    - b (float): Scaling factor used in the forward model.
    - e (float): Non-linearity point used in the forward model.

    Returns:
    - x (array): Reconstructed original data points.
    - flag (array): Flags indicating data points beyond the non-linearity or saturation.
    """
    x = (np.sqrt(-4*e*b+4*b*y+1)+2*e*b-1)/(2*b)
    x[x < e] = y[x < e]
    flag = np.zeros(len(y))
    beta = point_non_lin_important(b, e)
    flag[x > beta] = 1
    flag[y > sat] = 2
    return x, flag
    
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

def point_non_lin_important_2(b, non_lin_point):
    """
    Calculates a critical point related to the non-linearity in the threshold fitting model.

    Parameters:
    - b (float): Scaling factor for the quadratic part of the function.
    - non_lin_point (float): Point where non-linearity begins.

    Returns:
    - beta (float): Calculated critical point where significant non-linearity effects are observed.
    """
    e = non_lin_point
    max_non_linearity = 0.95
    beta = (-np.sqrt(1-max_non_linearity)*np.sqrt(1- (max_non_linearity + 4*b*non_lin_point)) + max_non_linearity + 2*b*non_lin_point - 1)/(2*b)
    beta_2 = ((2*b*e+max_non_linearity-1)-np.sqrt(4*b*e*(max_non_linearity-1)+(max_non_linearity-1)**2))/(2*b)
    return beta,beta_2

root_directory = '/home/olemar/Projects/Universitetet/MATS/MATS-L1-processing'
os.chdir(root_directory)

#%% Load data from non_linearity tests
channels = [1,2,3,4,5,6,7]
calibration_file = "calibration_data/calibration_data.toml" #only used to find where raw data shall be taken from
calibration_data = toml.load(calibration_file)

CCDitems = read_in_functions.read_CCDitems(calibration_data["primary_data"]["linearity"]["folder"])

#filter away possible data not to use from the dataset
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
            add_bias=True,
            remove_blanks=False,
        )

        new_df = pd.DataFrame({
        'manual': man_tot,
        'measured': inst_tot,
        'channel': channel_tot,
        'test': test_type,
        'signal_factor': signal_factor
        })

        df = pd.concat([df, new_df], ignore_index=True)

# #%% Generate non-linearity

test_types_non_linearity = ['row','col'] #test types to use for non-linearity estimation
non_linearity_data = pd.DataFrame(columns=["channel", "b", "e", "sat", "non_lin_important","sumwell_saturation","sumrow_saturation","pixel_saturation"]) #dataframe to hold non-linearity data for each channel

#Theoretical values for saturation
pixel_saturation = 150e3 / 34 # pixel saturate at 150 electrons
sumrow_saturation = pixel_saturation*4
sumwell_saturation = 32000 #Hard coded to reflect ADC-saturation


channels = [1,2,3,4,5,6,7]
data = pd.DataFrame()
for i,test_type in enumerate(test_types_non_linearity):
    data = pd.concat([data, df[(df['channel'].isin(channels)) & (df['test'] == test_type)]])
x = data['manual'].to_numpy()
y = data['measured'].to_numpy()
params = curve_fit(threshold_fit, x, y,p0=np.array([-0.0001,10000,30000]))

# [b,non_lin,sat] = params[0]
# non_lin_important = point_non_lin_important(b,non_lin)
# for channel in channels:
#     if channel in [5,6]:
#         pixel_saturation = 150e3 / 34 * 3/2
#     else:
#         pixel_saturation =  150e3 / 34 
#     new_row = {'channel': channel, 'b': b, 'e': non_lin, 'sat': sat, 'non_lin_important': non_lin_important, 'sumwell_saturation': sumwell_saturation, 'sumrow_saturation': sumrow_saturation, 'pixel_saturation': pixel_saturation }
#     non_linearity_data.loc[len(non_linearity_data)] = new_row

non_linearity_data = linearity.make_linearity(calibration_file)
non_linearity_data.to_csv('linearity.csv')

#%%
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

def get_color(category):
    if category in [1, 2, 3, 4]:
        return 'blue'  # Color for categories 1, 2, 3, 4
    elif category in [5, 6]:
        return 'green'  # Color for categories 5, 6
    elif category in [7]:
        return 'red'  # Color for categories 5, 6

#%%
colors = [get_color(cat) for cat in df["channel"][::100]]
plt.scatter(df['manual'][::100],df['measured'][::100],1,colors)
plt.plot(np.array([0,40000]),np.array([non_linearity_data['pixel_saturation'][0],non_linearity_data['pixel_saturation'][0]]),'blue')
plt.plot(np.array([0,40000]),np.array([non_linearity_data['pixel_saturation'][5],non_linearity_data['pixel_saturation'][5]]),'green')
plt.xlabel('True value')
plt.ylabel('Measured value')
# plt.xlim([0,20000])
# plt.ylim([0,10000])

plt.show()

# %%
y_fit = np.arange(0,35000)
e = non_linearity_data['e'][0]
sat = non_linearity_data['sat'][0]
b = non_linearity_data['b'][0]
x_fit,flag = threshold_fit_inv(y_fit,sumwell_saturation,b,e)
y_sat = threshold_fit(np.array([sat]),sat,b,e)[0]

# %%
plt.plot(np.array([0,40000]),np.array([0,40000]),'k-',linewidth=0.5)
plt.scatter(x,y,1,color='0.7',alpha=0.1)
plt.plot(x_fit[flag==0],y_fit[flag==0])
plt.plot(x_fit[flag==1],y_fit[flag==1])
plt.plot(x_fit[flag==2],y_fit[flag==2])
plt.plot(np.array([0,40000]),np.array([sumwell_saturation,sumwell_saturation]),'k:')# %%
plt.plot(np.array([e,e]),np.array([0,e]),'k--')
plt.plot(np.array([0,e]),np.array([e,e]),'k--')
plt.xlim([0,40000])
plt.ylim([0,40000])
plt.xlabel('estimated value')
plt.ylabel('measured value')
plt.legend(['data','valid', 'highly non-linear', 'saturated','saturation limit','start of non-linearity'])
plt.show()
# %%
df_pixel = df[df['test']=='exp']
df_uv = df_pixel[df_pixel['channel'].isin([5, 6])]
df_ir = df_pixel[df_pixel['channel'].isin([1,2,3,4,7])]
plt.scatter(df_uv['manual'][::10],df_uv['measured'][::10],1,color='m',alpha=0.1)
plt.scatter(df_ir['manual'][::10],df_ir['measured'][::10],1,color='r',alpha=0.1)
plt.plot(np.array([0,40000]),np.array([non_linearity_data['pixel_saturation'][5],non_linearity_data['pixel_saturation'][5]]),'m--')
plt.plot(np.array([0,40000]),np.array([non_linearity_data['pixel_saturation'][0],non_linearity_data['pixel_saturation'][0]]),'r--')
plt.plot(x_fit[flag==0],y_fit[flag==0])
plt.xlim([0,10000])
plt.ylim([0,10000])
plt.xlabel('estimated value')
plt.ylabel('measured value')
plt.legend(['uv data','ir data','uv pixel saturation','ir pixel saturation','non-linearity curve'])
plt.show()
# %%
