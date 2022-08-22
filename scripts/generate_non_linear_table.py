#%%
import json
import numpy as np
from matplotlib import pyplot as plt
from mats_l1_processing.L1_calibration_functions import CCD
from mats_l1_processing.L1_calibration_functions import inverse_model_real
from mats_l1_processing.read_in_functions import channel_num_to_str, add_and_rename_CCDitem_info
import pandas as pd
#%%
def gen_non_linear_table(CCDitem,calibrationfile):

    #Add stuff not in the non linear tables (but required for add_and_rename)
    CCDitem["read_from"] = 'rac'
    CCDitem["EXP Nanoseconds"] = 0
    CCDitem["BC"] = '[]'

    #CCDitem["channel"] = channel_num_to_str(CCDitem['CCDSEL'])
    #CCDitem["RID"] = 'CCD' + str(CCDitem['CCDSEL'])
    #CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)
    CCDitem = add_and_rename_CCDitem_info(CCDitem)
    CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)


    x_measured = np.arange(0,2**16-1,1)
    x_true = np.zeros(x_measured.shape)
    flag = np.zeros(x_measured.shape)
    for i in range(len(x_measured)):
        x_true[i],flag[i] = inverse_model_real(CCDitem,x_measured[i],method='Nelder-Mead')

    return x_true,x_measured,CCDitem,flag

#%%
def get_covariances(channel):
    df = pd.read_csv('calibration_data/linearity/final/covariance_col')
    cov_col = df.iloc[channel-1]

    df = pd.read_csv('calibration_data/linearity/final/covariance_row')
    cov_row = df.iloc[channel-1]

    df = pd.read_csv('calibration_data/linearity/final/covariance_exp')
    cov_exp = df.iloc[channel-1]

    return cov_col,cov_row,cov_exp


def gen_non_linear_error(CCDitem,calibrationfile):

    #Add stuff not in the non linear tables (but required for add_and_rename)
    CCDitem["read_from"] = 'rac'
    CCDitem["EXP Nanoseconds"] = 0
    CCDitem["BC"] = '[]'

    #CCDitem["channel"] = channel_num_to_str(CCDitem['CCDSEL'])
    #CCDitem["RID"] = 'CCD' + str(CCDitem['CCDSEL'])
    #CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)
    CCDitem = add_and_rename_CCDitem_info(CCDitem)
    CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)

    
    covariance_exp,covariance_row,covariance_col = get_covariances(CCDitem['CCDSEL'])
    
    return covariance_exp,covariance_row,covariance_col


directory = "calibration_data/linearity/tables/20220802/"
df = pd.read_csv(directory + "tables.csv")
items = df.to_dict("records")


'''
# %%
for i in range(len(items)):
    x_true_2,x_measured_2,CCDitem_2,flag_2 = gen_non_linear_table(items[i],'calibration_data/calibration_data.toml')
    table = np.array([x_true_2,flag_2,x_measured_2])

    np.save(CCDitem_2['Tablefile']+'.npy',table)
'''
#%%
for i in range(len(items)):
    gen_non_linear_error(items[i],'calibration_data/calibration_data.toml')
# %%
