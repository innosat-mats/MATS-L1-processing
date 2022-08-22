#%%
import pickle
import glob
import array_to_latex as a2l
import numpy as np
import pandas as pd

numpy_vars = {}
filelist = sorted( glob.glob('calibration_data/linearity/final/*.pkl'))
for file in filelist:
    with open(file, 'rb') as fp:
        numpy_vars[file] = pickle.load(fp)

keys = list(numpy_vars.keys())
df = pd.DataFrame(columns=['a','b','e','beta05','beta005','filename'])
for key in keys:
    prinout = numpy_vars[key].fit_parameters
    prinout.append(numpy_vars[key].get_measured_non_lin_important())
    prinout.append(numpy_vars[key].get_measured_saturation())
    prinout.append(key)
    df = df.append(pd.DataFrame([prinout],columns=['a','b','e','beta05','beta005','filename']))

a2l.to_ltx(df, frmt = '{:.5g}', arraytype = 'tabular')

# %%
