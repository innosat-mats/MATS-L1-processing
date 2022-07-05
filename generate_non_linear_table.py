#%%
import json
import numpy as np
from matplotlib import pyplot as plt
from mats_l1_processing.L1_calibration_functions import CCD
from mats_l1_processing.L1_calibration_functions import inverse_model_real

#%%
def gen_non_linear_table(CCDitemfile,calibrationfile):

    with open(CCDitemfile, 'r') as fp:
        CCDitem = json.load(fp)

    CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)
    
    x_measured = np.arange(0,6e3,1)
    x_true = np.zeros(x_measured.shape)
    for i in range(len(x_measured)):
        x_true[i] = inverse_model_real(CCDitem,i)

    return x_true,x_measured,CCDitem
# %%

x_true,x_measured,CCDitem = gen_non_linear_table('testdata/simulated_data/full_frame.json','tests/calibration_data_test.toml')
    # %%
