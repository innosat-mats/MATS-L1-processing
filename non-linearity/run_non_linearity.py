#%%
from database_generation.linearity import make_linearity
from mats_l1_processing.read_and_calibrate_all_files import main

# calibration_file = "tests/calibration_data_test.toml"
# #x_hat = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='exp',inverse=False)
# #x_hat = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='row',inverse=False)
# x_hat = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='col',inverse=False)

# print(x_hat)
# %% 

main("testdata/RacFiles_out/", "tests/calibration_data_test.toml")
