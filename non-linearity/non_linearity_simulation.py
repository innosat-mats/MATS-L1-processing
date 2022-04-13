#%%

import numpy as np
import scipy.optimize as opt

#%%
def row_sum(CCD,nrowbin):
    
    nrows = int(np.floor(CCD.shape[0]/nrowbin)) 
    CCD_binned = np.zeros((nrows,CCD.shape[1]))
    for i in range(nrows):
        CCD_binned[i,:] = np.sum(CCD[i*nrowbin:i*nrowbin+nrowbin,:],axis=0)

    return np.mean(CCD_binned,axis=0)

def col_sum(CCD,ncolbin):
    
    ncols = int(np.floor(CCD.shape[0]/ncolbin)) 
    CCD_binned = np.zeros((ncols,1))
    for i in range(ncols):
        CCD_binned[i,:] = np.sum(CCD[i*ncolbin:i*ncolbin+ncolbin],axis=0)
    
    return np.mean(CCD_binned,axis=0) 

def row_sum_v1(CCD,nrowbin):
    return CCD.reshape(-1,nrowbin,CCD.shape[-1]).sum(1)

def col_sum_v1(CCD,ncolbin):
    return CCD.reshape(CCD.shape[0],-1,ncolbin).sum(2)

def transfer_function(value_in,a,b):
    return a*value_in+b

def sum_well(CCD,ncolbin,a,b):
    return transfer_function(col_sum(CCD,ncolbin),a,b)

def shift_register(CCD,nrowbin,a,b):
    return transfer_function(row_sum(CCD,nrowbin),a,b)

def total_model(CCD,nrowbin,ncolbin,x):
    return sum_well(shift_register(transfer_function(CCD,x[0],x[1]),nrowbin,x[2],x[3]),ncolbin,x[4],x[5])

def forward_model(x):
    nrowbin = np.arange(1,5,dtype=int)
    ncolbin = np.arange(1,5,dtype=int)
    nrow = 2
    ncol = 2
    rows_tot = nrowbin.max()*nrow
    cols_tot = ncolbin.max()*ncol
    value = 5

    CCD = np.ones((rows_tot,cols_tot))*value
    #print(CCD)

    y = np.zeros([len(nrowbin),len(ncolbin)])
    for i in range(len(nrowbin)):
        for j in range(len(ncolbin)):
            y[i,j] = total_model(CCD,nrowbin[i],ncolbin[j],x)[0]
            
    return y

def forward_model_real():
    non_linearity = np.load(
        '../calibration_data/linearity/'
        + "linearity_"
        + "1"
        + ".npy"
    )

    return None


def optimize_function(x):
    x_true = np.array([1,10,1,0,1,0])
    x2 = np.append(x,[1,0,1,0])
    y_true = forward_model(x_true)
    
    y_iter = forward_model(x2)
    
    return np.linalg.norm(y_true.flatten()-y_iter.flatten())
    

#non_linearity = np.load(
#    '../calibration_data/linearity/'
#    + "linearity_"
#    + "1"
#    + ".npy"
#)
#np.polyval(non_linearity,1000)

#%%

x0 = np.array([1,0])
x_hat = opt.minimize(optimize_function, x0,method='Nelder-Mead')
# %%
