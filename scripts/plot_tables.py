#%%
import numpy as np
from matplotlib import pyplot as plt
import glob
from matplotlib.pyplot import cm


#%%
numpy_vars = {}

files = sorted(glob.glob('../calibration_data/linearity/tables/20220802/highres*.npy'))

for i in range(len(files)):
    
    numpy_vars[i] = np.load(files[i])

# %%
color = cm.rainbow(np.linspace(0, 1, 7))

for i in [0,1,2,3,4,5]:

    I = np.where(numpy_vars[i][1,:]==0)
    plt.plot(numpy_vars[i][0,I[0]],numpy_vars[i][2,I[0]],c=color[i+1],label='channel ' + str(i+1))
    I = np.where(numpy_vars[i][1,:]==1)
    plt.plot(numpy_vars[i][0,I[0]],numpy_vars[i][2,I[0]],c=color[i+1],linestyle='--',label=None)
    I = np.where(numpy_vars[i][1,:]==3)
    plt.plot(numpy_vars[i][0,I[0]],numpy_vars[i][2,I[0]],c=color[i+1],linestyle=':',label=None)

plt.plot(np.array([0,100000]),np.array([0,100000]),'k--')
plt.legend()
plt.xlabel('true counts')
plt.ylabel('measured counts')
plt.ylim(0,65e3)
plt.xlim(0,90e3)
plt.grid(True)
plt.savefig("linearity_sciencemode" + ".png", dpi=600)
# %%
