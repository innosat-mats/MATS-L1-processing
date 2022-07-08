#%%
import database_generation
from database_generation.linearity import make_linearity
import numpy as np
from mats_l1_processing.read_and_calibrate_all_files import main

#%%
calibration_file = "calibration_data/linearity/calibration_data_col.toml"
poly_or_spline = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='col',inverse=False)

#%%
calibration_file = "calibration_data/linearity/calibration_data_col.toml"
poly_or_spline = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='row',inverse=False)

#%%
calibration_file = "calibration_data/linearity/calibration_data_col.toml"
poly_or_spline = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='exp',inverse=False)
# print(x_hat)
# %% 

#main("testdata/RacFiles_out/", "tests/calibration_data_test.toml")

