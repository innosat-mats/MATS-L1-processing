#%%
import database_generation
from database_generation.linearity import make_linearity
from database_generation.linearity import get_threshold
import numpy as np
from mats_l1_processing.read_and_calibrate_all_files import main

calibration_file = "tests/calibration_data_test.toml"
poly_or_spline = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='exp',inverse=False)
#thresholds = np.array([])
#for i in range(poly_or_spline.shape[0]):
#    thresholds.append(get_threshold(poly_or_spline[i]))


poly_or_spline = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='row',inverse=False)


poly_or_spline = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='col',inverse=False)

# print(x_hat)
# %% 

#main("testdata/RacFiles_out/", "tests/calibration_data_test.toml")
