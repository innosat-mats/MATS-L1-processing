#%%
import matplotlib.pylab as plt
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import LSQBivariateSpline
import pickle

#%%

def make_spline_table(FM, temps, bits, name):
    #plt.figure()
    #plt.pcolor(temps,bits,FM.T)
    tmesh,bmesh=np.meshgrid(temps,bits)
    tknots=np.linspace(min(temps),max(temps),10)
    bknots=np.linspace(min(bits),max(bits),10)
    spline=LSQBivariateSpline(tmesh.flatten(),bmesh.flatten(),FM.T.flatten(),tknots,bknots)    
    pickle.dump(spline, open(name + ".pkl", "wb")) 


    #plt.figure()
    #plt.pcolor(temps[::10],bits[::10],spline(temps[::10],bits[::10]).T)

    return (spline(temps[::10],bits[::10]) - FM[::10,::10])/FM[::10,::10]

#%%
data=loadmat('/home/olemar/Projects/Universitetet/MATS/MATS-L1-processing/calibration_data/photometers/AlbedoFM_Calib_Tdep_Radiance_vs_bits.mat')

# %%
name = "SignFM1_Rad_raw"
FM=data[name]
temps=data['Temperatur'].squeeze()[:]
bits=data['bitar'].squeeze()
rms1 = make_spline_table(FM, temps, bits, name)
print(rms1)

#%%
name = "SignFM2_Rad_raw"
FM=data[name][:,2:]
temps=data['Temperatur'].squeeze()[:]
bits=data['bitar'].squeeze()[2:]
rms2 = make_spline_table(FM, temps, bits, name)
print(rms2)

# %%
