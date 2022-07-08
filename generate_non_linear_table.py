#%%
import json
import numpy as np
from matplotlib import pyplot as plt
from mats_l1_processing.L1_calibration_functions import CCD
from mats_l1_processing.L1_calibration_functions import inverse_model_real
from mats_l1_processing.read_in_functions import channel_num_to_str
#%%
def gen_non_linear_table(CCDitemfile,calibrationfile,channel=1):

    with open(CCDitemfile, 'r') as fp:
        CCDitem = json.load(fp)

    CCDitem["channel"] = channel_num_to_str(channel)
    CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)
    
    x_measured = np.arange(0,6e3,1)
    x_test = np.zeros(x_measured.shape)
    x_true = np.zeros(x_measured.shape)
    flag = np.zeros(x_measured.shape)
    for i in range(len(x_measured)):
        x_true[i],flag[i] = inverse_model_real(CCDitem,i,method='Nelder-Mead')
        x_test[i] = CCDitem["CCDunit"].non_linearity_sumwell.get_measured_value(CCDitem["CCDunit"].non_linearity_sumrow.get_measured_value(CCDitem["CCDunit"].non_linearity_pixel.get_measured_value(x_true[i])))      

    return x_true,x_measured,x_test,CCDitem,flag

# %%
channel = 5
#x_true,x_measured,x_test,CCDitem,flag = gen_non_linear_table('testdata/simulated_data/full_frame.json','calibration_data/calibration_data.toml',channel)

# %%
x_true_2,x_measured_2,x_test_2,CCDitem_2,flag_2 = gen_non_linear_table('testdata/simulated_data/full_frame.json','calibration_data/calibration_data_linear.toml',channel)


# %%
#errorvalue=0
#plt.plot(x_true,x_measured,'o')
#plt.plot(x_true[flag==errorvalue],x_measured[flag==errorvalue])


# %%
errorvalue=0
plt.plot(x_true_2,x_measured_2,'o')
plt.plot(x_true_2[flag_2==errorvalue],x_measured_2[flag_2==errorvalue])
# %%
