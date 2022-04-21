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

def row_sum_v2(CCD,nrowbin):
    return CCD.reshape(-1,nrowbin,CCD.shape[-1]).sum(1)

def col_sum_v1(CCD,ncolbin):
    return CCD.reshape(CCD.shape[0],-1,ncolbin).sum(2)

def transfer_function(value_in,poly):
    return np.polyval(poly,value_in)

def sum_well(CCD,ncolbin,poly):
    return transfer_function(col_sum(CCD,ncolbin),poly)

def shift_register(CCD,nrowbin,poly):
    return transfer_function(row_sum(CCD,nrowbin),poly)

def single_pixel(CCD,texp,poly):
    return transfer_function(CCD*texp,poly)

def total_model(CCD,nrowbin,ncolbin,texp,p):
    return sum_well(shift_register(single_pixel(CCD,texp,p[0]),nrowbin,p[1]),ncolbin,p[2])

def forward_model(x):
    nrowbin = np.arange(1,10,dtype=int)
    ncolbin = np.arange(1,5,dtype=int)
    texp = np.arange(1,5,dtype=int)
    nrow = 2
    ncol = 2
    rows_tot = nrowbin.max()*nrow
    cols_tot = ncolbin.max()*ncol
    value = 1

    CCD = np.ones((rows_tot,cols_tot))*value
    #print(CCD)

    y = np.zeros([len(nrowbin),len(ncolbin),len(texp)])
    for i in range(len(nrowbin)):
        for j in range(len(ncolbin)):
            for k in range(len(texp)):
                y[i,j,k] = total_model(CCD,nrowbin[i],ncolbin[j],texp[k],x)[0]
            
    return y


def optimize_function(x,x_true=[0.5,8,1,10,0.8,12]):
    y_true = forward_model(x_true)
    
    x2 = x
    y_iter = forward_model(x2)
    
    return np.linalg.norm(y_true.flatten()-y_iter.flatten())
    
    

def total_model_scalar(x,nrowbin,ncolbin,texp):
    cal_consts = []
    cal_consts.append(np.load(
       '../calibration_data/linearity/'
       + "linearity_"
       + "1_exp"
       + ".npy"))
    # cal_consts.append(np.array([0, 1, 0]))
    cal_consts.append(np.array([0, 1, 0]))
    cal_consts.append(np.array([0, 1, 0]))

    CCD = np.ones((nrowbin,ncolbin))*x
    return total_model(CCD,nrowbin,ncolbin,texp,cal_consts)

def optimize_function_scalar(x,nrowbin,ncolbin,texp,value):
    #x is true value, y is measured value
    y_model = total_model_scalar(x,nrowbin,ncolbin,texp)
    
    return np.abs(y_model-value)

def inverse_model_real(nrowbin,ncolbin,texp,value):
    #non_linearity = np.load(
    #    '../calibration_data/linearity/'
    #    + "linearity_"
    #    + "1"
    #    + ".npy"
    #)

    x = opt.minimize_scalar(optimize_function_scalar,args=(nrowbin,ncolbin,texp,value))
    return x


#%%
'''
x_true=np.array([0.5,8,1,10,0.8,12])
x0 = np.array([1,0,1,0,1,0])

x_hat = opt.minimize(optimize_function, x0,args=(x_true),method='SLSQP')

x_hat.x-x_true
'''
# %%

cal_consts = []
cal_consts.append(np.load(
    '../calibration_data/linearity/'
    + "linearity_"
    + "1_exp"
    + ".npy"))
# cal_consts.append(np.array([0, 1, 0]))
cal_consts.append(np.array([0, 1, 0]))
cal_consts.append(np.array([0, 1, 0]))
nrowbin = 1
ncolbin = 1
x = 8000
texp = 1
CCD = np.ones((nrowbin,ncolbin))/(nrowbin*ncolbin)*x
print(total_model(CCD,nrowbin,ncolbin,texp,cal_consts))

# %%

# %%
