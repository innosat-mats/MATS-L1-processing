#%%
from database_generation.linearity import threshold_fit_2
from matplotlib import pyplot as plt
import numpy as np
import glob
from matplotlib.pyplot import cm

files = sorted(glob.glob('../*.npz'))
color = cm.rainbow(np.linspace(0, 1, 7))

res = np.array([])
for i in range(len(files)):
    
    a = np.load(files[i])
    x = a.f.arr_0
    y = a.f.arr_1
    y_f = threshold_fit_2(x,a.f.arr_2[0],a.f.arr_2[1],a.f.arr_2[2])
    plt.plot(x,y,'.',c=color[i],alpha=0.1)
    plt.plot(x,y_f,'--',c=color[i])
    #plt.plot(x,(y-y_f)/y)
    res = np.append(res,(y-y_f)/y_f)
# %%

plt.show()