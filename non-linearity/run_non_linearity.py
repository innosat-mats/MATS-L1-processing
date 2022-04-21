#%%
from database_generation.linearity import make_linearity

calibration_file = "tests/calibration_data_test.toml"
#x_hat = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='exp',inverse=False)
#x_hat = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='row',inverse=False)
x_hat = make_linearity([1,2,3,4,5,6], calibration_file, plot=True,exp_type='col',inverse=False)

print(x_hat)
# %% 
