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
    x_test = np.zeros(x_measured.shape)
    x_true = np.zeros(x_measured.shape)
    for i in range(len(x_measured)):
        x_true[i] = inverse_model_real(CCDitem,i)
        x_test[i] = CCDitem["CCDunit"].non_linearity_sumwell.get_measured_value(CCDitem["CCDunit"].non_linearity_sumrow.get_measured_value(CCDitem["CCDunit"].non_linearity_pixel.get_measured_value(x_true[i])))      

    return x_true,x_measured,x_test,CCDitem
# %%
x_true,x_measured,x_test,CCDitem = gen_non_linear_table('testdata/simulated_data/full_frame.json','calibration_data/calibration_data.toml')

# %%
plt.plot(x_true,x_measured,x_true,x_test,'.')

# %%
x_true_2,x_measured_2,x_test_2,CCDitem_2 = gen_non_linear_table('testdata/simulated_data/full_frame.json','calibration_data/calibration_data_linear.toml')

# %%
plt.plot(x_true_2,x_measured_2,x_true_2,x_measured_2,'.')
# %%
plt.plot(x_true_2,x_measured_2,x_true,x_measured)
# %%
