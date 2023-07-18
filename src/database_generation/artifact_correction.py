#%% Import modules
#%matplotlib qt5
import datetime as DT
from tqdm import tqdm 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import warnings
from mats_utils.rawdata.read_data import read_MATS_data



#%%
im_ref_def = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0.]])

def_sampling_rate = timedelta(seconds=2) # sampling rate of the NADIR images
def_pix_shift = 2.6 # pixel shift along the y axis between 2 consecutive images. For a sampling time of 2s it is equivalent to 2.6 pixels

def nadir_shift(im_ref,sampling_rate=def_sampling_rate,pix_shift=def_pix_shift):
    """
    Function calculating which pixels in the nadir images should be taken as reference for the correction of the other pixels. The Nadir sensors
    sees the same feature on the ground in consecutive images. By choosing pixels as reference, they can be compared to pixels to be corrected in
    other images taken just before or after.
    
    Arguments:
        im_ref : np.array[float] (shape = (a,b))
            array representing nadir images, pixels with value 1 are taken as reference pixels, pixels with a value of 0 have to be corrected
        sampling_rate : timedelta
            time difference between 2 nadir exposures. Default value is 2s
        pix_shift : float
            pixel shift along the y axis between 2 consecutive images. For a sampling time of 2s it is equivalent to 2.6 pixels

    Returns:
        im_shift : np.array[float] (shape = (a,b)))  
            each pixel has a value corresponding to the number of images it is shifted compared to the corresponding reference pixel 
        pix_ref : np.array[float] (shape = (a,b,2))
            for each pixel, the indices of the corresponding reference pixel are given. The first index is the row index, the second the column index
    """
    
    a,b = np.shape(im_ref)
    im_shift = np.zeros((a,b))
    pix_ref = np.zeros((a,b,2))
    

    for y_cor in range(a): # looping on the rows
        for x_cor in range(b): # looping on the columns
            if im_ref[y_cor,x_cor] == 1:
                im_shift[y_cor,x_cor] = None
                pix_ref[y_cor,x_cor,0] = None
                pix_ref[y_cor,x_cor,1] = None
            else:
                min_err = 1000
                x_ref = x_cor
                for y_ref in range(a):
                    if im_ref[y_ref,x_ref] == 1:
                        if abs((y_cor-y_ref)%pix_shift) < min_err:
                            min_err = abs((y_cor-y_ref)%pix_shift)
                            im_shift[y_cor,x_cor] = (y_cor-y_ref)//pix_shift
                            pix_ref[y_cor,x_cor,0] = y_ref
                            pix_ref[y_cor,x_cor,1] = x_ref
    return im_shift, pix_ref

                    




def azimuth_bias_mask(ccditems,bias_threshold,az_list=None,sampling_rate=def_sampling_rate,pix_shift=def_pix_shift,show_plots=False):
    """
    Function creating correction masks dependant on solar azimuth angles. The mask
    creation follows the same rules as in the function nadir_mask. The bias and R2 
    masks are created for each azimuth angle intervall.     
   
    Arguments:
        ccditems : Panda dataframe
            dataframe containing the images
        bias_threshhold : float
            all bias values smaller than this value are not taken into account (set to zero)
        az_list : list of float
            list of azimuth value. The regression is made on azimuth angle intervalls with the 
            given angles as center points 
        sampling_rate : timedelta
            sampling rate of the NADIR images
        pix_shift : float
            pixel shift along the y axis between 2 consecutive images. For a sampling time of 2s it is equivalent to 2.6 pixels

        

    Returns:
        azimuth_masks : Pandas dataframe
            'bias_mask' : np.array[float]
                thresholded bias for each pixel, has the same shape as the images
            'R2_mask' : np.array[float]
                R2 value for each pixel, has the same shape as the images
            'azimuth' : float
                center value of each azimuth intervall. This might not be the case 
            
    """
    # only using NADIR images to set the mask values
    ccditems = ccditems[ccditems['CCDSEL'] == 7]
    n = len(ccditems)
    if ccditems.iloc[0]['DataLevel'] == 'L1B' :
        im_key = 'ImageCalibrated'
        print('DataLevel L1b')
    elif ccditems.iloc[0]['DataLevel'] == 'L1A' :
        im_key = 'IMAGE'
        print('DataLevel L1a')
    else :
        warnings.warn('DataLevel not recognized (should be L1A or L1B)')

    a,b = np.shape(ccditems.iloc[0][im_key])

       
    im_points = np.zeros((n,a,b)) # array containing all the pixel values
    NADIR_AZ = np.array(ccditems['nadir_az']) # list of nadir solar azimuth angles
    EXP_DATE = np.array(ccditems['EXPDate']) # list of exposition dates
    
    IM_BIAS = [] # list of bias masks
    IM_R2 = []  # list of R2 masks
   
    # filling the several arrays
    for i in range(n):
        ccditem = ccditems.iloc[i]        
        im = ccditem[im_key]
        im_points [i,:,:] = im       
    

    # if no azimuth list is given, the azimuth angles are equally distributed between the min and max angles
    if az_list is None:
        az_list = np.linspace(min(NADIR_AZ),max(NADIR_AZ),100)

    # intermediate function for regression (only the bias is optimized)
    def func(X,b):
        return X+b
    
    im_shift, pix_ref = nadir_shift(im_ref_def,sampling_rate,pix_shift)

    for i in tqdm(range(len(az_list)),desc='computing bias values'): # looping on the azimuth angle intervals
        
        # lower interval limit
        if i == 0:
            az_min = min(NADIR_AZ) 
        else :
            az_min = (az_list[i-1] + az_list[i])/2.0

        # higher interval limit
        if i == len(az_list)-1:
            az_max = max(NADIR_AZ) 
        else :
            az_max = (az_list[i+1] + az_list[i])/2.0 
        
        
        # regression on the selected images
        im_R2 = np.ones_like(im_points[0,:,:]) # R2 values for each pixel
        im_bias = np.zeros_like(im_points[0,:,:]) # bias value for each pixel

        
        for x_cor in range(b):
            for y_cor in range(a):
                
                step = im_shift[y_cor,x_cor] # images offset between the reference and corrected pixel can be positive (image to be corrected exposed after the reference image) or negative (other way around)
                y_ref = pix_ref[y_cor,x_cor,0] # column index of the pixel taken as reference
                x_ref = pix_ref[y_cor,x_cor,1] # row index of the pixel taken as reference

                if not np.isnan(step): # if the pixel is not a reference pixel 
                    step = int(step)
                    y_ref = int(y_ref)
                    x_ref = int(x_ref)
                    #print(x_cor,y_cor,step,y_ref)

                    X = [] # list of reference pixel values
                    Y = [] # list of artifact pixel values
                    REG_AZ = [] # list of azimuth angles of the images

                    az_index_cor = (az_min < NADIR_AZ) & (NADIR_AZ < az_max) # indices for the images with in the given azimuth angle interval (the pixels to be corrected are taken from these images)
                    if step > 0:
                        cor_indexes = np.arange(n)[az_index_cor][:-step] # indices of the corresponding reference images where the reference pixel is taken
                    else:
                        cor_indexes = np.arange(n)[az_index_cor][step:] # indices of the corresponding reference images where the reference pixel is taken


                    if len(cor_indexes) > 0 : 
                        for art_ind in cor_indexes: # iterating over the images (image index of the artifact pixel)
                            ref_ind = art_ind + step # image index of the reference 
                            #check if the image taken as reference is indeed taken the right amount of time after the other image
                            if (EXP_DATE[art_ind] - EXP_DATE[ref_ind] - step*sampling_rate) < timedelta(seconds= 0.01):
                                X.append(im_points[ref_ind,y_ref,x_ref])
                                Y.append(im_points[art_ind,y_cor,x_cor])
                                REG_AZ.append(NADIR_AZ[art_ind])

                    if len(X) + len(Y) > 0:
                    # linear regression, the slope is set to 1 
                        fit_param, cov = curve_fit(func,X,Y)
                        abs_err = Y-func(X,fit_param[0])
                        rsquare = 1.0 - (np.var(abs_err)/np.var(Y))
                        intercept = fit_param[0]
                        im_R2[y_cor,x_cor] = rsquare
                        im_bias[y_cor,x_cor] = intercept
                        if show_plots and x_cor==15 and y_cor==10:
                            slope = 1.0
                            plt.figure()
                            plt.scatter(X,Y,c=REG_AZ)
                            plt.plot(X,intercept + X,color='red')
                            plt.title(f"Offset of {step} images. Slope = {slope:.3f}, intercept = {intercept:.1f}, R**2 = {rsquare:.3f}")
                            # plt.title(f"Artifact correlation, offset of {step} images. Slope = {slope:.3f}, intercept = {intercept:.1f}, R**2 = {rsquare:.3f}")
                            plt.xlabel(f'Reference pixel ({x_ref},{y_ref})')
                            plt.ylabel(f'Corrected pixel ({x_cor},{y_cor})')
                            plt.colorbar(label='nadir azimuth')
                            plt.show() 

                    else : 
                        im_R2[y_cor,x_cor] = None
                        im_bias[y_cor,x_cor] = None    
                               
        
        # thresholding the correction
        mask = im_bias>bias_threshold       
        bias_mask = im_bias * mask
        R2_mask = im_R2 * mask
            
        IM_BIAS.append(bias_mask)
        IM_R2.append(R2_mask)
        
    
    # creating dataframe
    azimuth_masks = pd.DataFrame({'bias_mask': IM_BIAS,
                                 'R2_mask': IM_R2,
                                 'azimuth': az_list})
    return azimuth_masks

def save_masks(azimuth_masks,filename):
    """
    Function saving the masks created with the function azimuth_bias_mask in a .pkl file
    """
    azimuth_masks_noR2 = azimuth_masks.drop(labels='R2_mask',axis=1,inplace=True)
    azimuth_masks_noR2.to_pickle(filename)    
    return  

def reg_analysis(ccditems,x_cor,y_cor,az_min,az_max,sampling_rate=def_sampling_rate,pix_shift=def_pix_shift):
    """
    Function plotting the regression between a given pixel and the corresponding reference pixel

    Arguments:
        x_cor : int
            column index of the pixel to be analyzed
        y_cor : int
            row index of the pixel to be analyzed
        ccditems : Panda dataframe
            dataframe containing the images
        az_min : float
            lower limit of the azimuth angle interval
        az_max : float
            higher limit of the azimuth angle interval
        sampling_rate : timedelta
            sampling rate of the NADIR images. Default value is def_sampling_rate
        pix_shift : float
            pixel shift along the y axis between 2 consecutive images. For a sampling time of 2s it is equivalent to 2.6 pixels. Default value is def_pix_shift
        
    Returns:
        None
    """
    # only using NADIR images to set the mask values
    ccditems = ccditems[ccditems['CCDSEL'] == 7]
    n = len(ccditems)

    if ccditems.iloc[0]['DataLevel'] == 'L1B' :
        im_key = 'ImageCalibrated'
        print('DataLevel L1b')
    elif ccditems.iloc[0]['DataLevel'] == 'L1A' :
        im_key = 'IMAGE'
        print('DataLevel L1a')
    else :
        warnings.warn('DataLevel not recognized (should be L1A or L1B)')

    a,b = np.shape(ccditems.iloc[0][im_key])

    pixel_shift = 2.6 # pixel shift along the y axis between 2 consecutive images. For a sampling time of 2s it is equivalent to 2.6 pixels
    
    im_points = np.zeros((n,a,b)) # array containing all the pixel values
    NADIR_AZ = np.array(ccditems['nadir_az']) # list of nadir solar azimuth angles
    EXP_DATE = np.array(ccditems['EXPDate']) # list of exposition dates
   
    # filling the several arrays
    for i in range(n):
        ccditem = ccditems.iloc[i]        
        im = ccditem[im_key]
        im_points [i,:,:] = im       

    # intermediate function for regression (only the bias is optimized)
    def func(X,b):
        return X+b
    
    im_shift, pix_ref = nadir_shift(im_ref_def,sampling_rate,pix_shift)        
        
    step = im_shift[y_cor,x_cor] # images offset between the reference and corrected pixel
    y_ref = pix_ref[y_cor,x_cor,0] # column index of the pixel taken as reference
    x_ref = pix_ref[y_cor,x_cor,1] # row index of the pixel taken as reference

    if not np.isnan(step): # if the pixel is not a reference pixel 
        step = int(step)
        y_ref = int(y_ref)
        x_ref = int(x_ref)
        #print(x_cor,y_cor,step,y_ref)

        X = [] # list of reference pixel values
        Y = [] # list of artifact pixel values
        REG_AZ = [] # list of azimuth angles of the images

        az_index_cor = (az_min < NADIR_AZ) & (NADIR_AZ < az_max) # indices for the images with in the given azimuth angle interval (the pixels to be corrected are taken from these images)
        if step > 0:
            cor_indexes = np.arange(n)[az_index_cor][:-step] # indices of the corresponding reference images where the reference pixel is taken
        else:
            cor_indexes = np.arange(n)[az_index_cor][step:] # indices of the corresponding reference images where the reference pixel is taken


        if len(cor_indexes) > 0 : 
            for art_ind in cor_indexes: # iterating over the images (image index of the artifact pixel)
                ref_ind = art_ind + step # image index of the reference 
                #check if the image taken as reference is indeed taken the right amount of time after the other image
                if (EXP_DATE[art_ind] - EXP_DATE[ref_ind] - step*sampling_rate) < timedelta(seconds= 0.01):
                    X.append(im_points[ref_ind,y_ref,x_ref])
                    Y.append(im_points[art_ind,y_cor,x_cor])
                    REG_AZ.append(NADIR_AZ[art_ind])

        if len(X) + len(Y) > 0:
        # linear regression, the slope is set to 1 
            fit_param, cov = curve_fit(func,X,Y)
            abs_err = Y-func(X,fit_param[0])
            rsquare = 1.0 - (np.var(abs_err)/np.var(Y))
            intercept = fit_param[0]
            slope = 1.0
            plt.figure()
            plt.scatter(X,Y,c=REG_AZ)
            plt.plot(X,intercept + X,color='red')
            plt.title(f"Offset of {step} images. Slope = {slope:.3f}, intercept = {intercept:.1f}, R**2 = {rsquare:.3f}")
            # plt.title(f"Artifact correlation, offset of {step} images. Slope = {slope:.3f}, intercept = {intercept:.1f}, R**2 = {rsquare:.3f}")
            plt.xlabel(f'Reference pixel ({x_ref},{y_ref})')
            plt.ylabel(f'Corrected pixel ({x_cor},{y_cor})')
            plt.colorbar(label='nadir azimuth')
            plt.show()  


    return 


def bias_analysis_angle(x_cor,y_cor,ccditems=None,azimuth_masks=None,az_list=None,sampling_rate=def_sampling_rate,pix_shift=def_pix_shift):
    """
    Function plotting the bias and R2 values for a given pixel in the different azimuth angle intervals

    Arguments:
        x_cor : int
            column index of the pixel to be analyzed
        y_cor : int
            row index of the pixel to be analyzed
        ccditems : Panda dataframe
            dataframe containing the images. It is used only if no azimuth_masks are given
        azimuth_masks : Pandas dataframe
            dataframe containing the masks created with the function azimuth_bias_mask. If the value is None, 
            the masks are created with the ccditems dataframe
        az_list : list of float
            list of azimuth value. The regression is made on azimuth angle intervalls with the
            given angles as center points
        sampling_rate : timedelta
            sampling rate of the NADIR images. Default value is def_sampling_rate
        pix_shift : float
            pixel shift along the y axis between 2 consecutive images. For a sampling time of 2s it is equivalent 
            to 2.6 pixels. Default value is def_pix_shift
        
    Returns:
        None
    """


    if type(azimuth_masks) == type(None): # if no mask is given, the masks are created
        # only using NADIR images to set the mask values
        ccditems = ccditems[ccditems['CCDSEL'] == 7]
        n = len(ccditems)

        if ccditems.iloc[0]['DataLevel'] == 'L1B' :
            im_key = 'ImageCalibrated'
            print('DataLevel L1b')
        elif ccditems.iloc[0]['DataLevel'] == 'L1A' :
            im_key = 'IMAGE'
            print('DataLevel L1a')
        else :
            warnings.warn('DataLevel not recognized (should be L1A or L1B)')

        a,b = np.shape(ccditems.iloc[0][im_key])

        im_points = np.zeros((n,a,b)) # array containing all the pixel values
        NADIR_AZ = np.array(ccditems['nadir_az']) # list of nadir solar azimuth angles
        EXP_DATE = np.array(ccditems['EXPDate']) # list of exposition dates
        
        IM_BIAS = [] # list of bias masks
        IM_R2 = []  # list of R2 masks
    
        # filling the several arrays
        for i in range(n):
            ccditem = ccditems.iloc[i]        
            im = ccditem[im_key]
            im_points [i,:,:] = im       
        

        # if no azimuth list is given, the azimuth angles are equally distributed between the min and max angles
        if az_list is None:
            az_list = np.linspace(min(NADIR_AZ),max(NADIR_AZ),100)

        # intermediate function for regression (only the bias is optimized)
        def func(X,b):
            return X+b
        
        im_shift, pix_ref = nadir_shift(im_ref_def,sampling_rate,pix_shift)
            
        for i in tqdm(range(len(az_list)),desc='bias analysis'): # looping on the azimuth angle intervals
            
            # lower interval limit
            if i == 0:
                az_min = min(NADIR_AZ) 
            else :
                az_min = (az_list[i-1] + az_list[i])/2.0

            # higher interval limit
            if i == len(az_list)-1:
                az_max = max(NADIR_AZ) 
            else :
                az_max = (az_list[i+1] + az_list[i])/2.0 
            
            
            # regression on the selected images
            im_R2 = np.ones_like(im_points[0,:,:]) # R2 values for each pixel
            im_bias = np.zeros_like(im_points[0,:,:]) # bias value for each pixel


            step = int(im_shift[y_cor,x_cor]) # images offset between the reference and corrected pixel
            y_ref = int(pix_ref[y_cor,x_cor,0]) # column index of the pixel taken as reference
            x_ref = int(pix_ref[y_cor,x_cor,1]) # row index of the pixel taken as reference

            #print(x_cor,y_cor,step,y_ref)

            X = [] # list of reference pixel values
            Y = [] # list of artifact pixel values

            az_index_cor = (az_min < NADIR_AZ) & (NADIR_AZ < az_max) # indices for the images with in the given azimuth angle interval (the pixels to be corrected are taken from these images)
            if step > 0:
                cor_indexes = np.arange(n)[az_index_cor][:-step] # indices of the corresponding reference images where the reference pixel is taken
            else:
                cor_indexes = np.arange(n)[az_index_cor][step:] # indices of the corresponding reference images where the reference pixel is taken


            if len(cor_indexes) > 0 : 
                for art_ind in cor_indexes: # iterating over the images (image index of the artifact pixel)
                    ref_ind = art_ind + step # image index of the reference 
                    #check if the image taken as reference is indeed taken the right amount of time after the other image
                    if (EXP_DATE[art_ind] - EXP_DATE[ref_ind] - step*sampling_rate) < timedelta(seconds= 0.01):
                        X.append(im_points[ref_ind,y_ref,x_ref])
                        Y.append(im_points[art_ind,y_cor,x_cor])

            if len(X) + len(Y) > 0:
            # linear regression, the slope is set to 1 
                fit_param, cov = curve_fit(func,X,Y)
                abs_err = Y-func(X,fit_param[0])
                rsquare = 1.0 - (np.var(abs_err)/np.var(Y))
                intercept = fit_param[0]
                im_R2[y_cor,x_cor] = rsquare
                im_bias[y_cor,x_cor] = intercept
            else : 
                im_R2[y_cor,x_cor] = None
                im_bias[y_cor,x_cor] = None               
            
                            
            IM_BIAS.append(im_bias)
            IM_R2.append(im_R2)
            
        
        # creating dataframe
        azimuth_masks = pd.DataFrame({'bias_mask': IM_BIAS,
                                    'R2_mask': IM_R2,
                                    'azimuth': az_list})
        
    plt.figure()
    for i in range(len(azimuth_masks)):
        plt.scatter(azimuth_masks['azimuth'].iloc[i],azimuth_masks['bias_mask'].iloc[i][y_cor,x_cor],c='C0')
    plt.xlabel('Nadir azimuth angle (deg)')
    plt.ylabel('Bias value')
    plt.title(f'Bias value (Pixel ({x_cor},{y_cor}))')
    plt.show()

    plt.figure()    
    for i in range(len(azimuth_masks)):
        plt.scatter(azimuth_masks['azimuth'].iloc[i],azimuth_masks['R2_mask'].iloc[i][y_cor,x_cor],c='C0')
    plt.xlabel('Nadir azimuth angle (deg)')
    plt.ylabel('R squared value')
    plt.ylim(bottom=0.5)
    plt.title(f'R squared value (Pixel ({x_cor},{y_cor}))')
    plt.show()

    return

def bias_analysis_histo(az_min,az_max,ccditems=None,azimuth_masks=None,sampling_rate=def_sampling_rate,pix_shift=def_pix_shift):
    """
    Function plotting the bias and R2 values for the whole image in the given azimuth angle interval

    Arguments:
        x_cor : int
            column index of the pixel to be analyzed
        y_cor : int
            row index of the pixel to be analyzed
        ccditems : Panda dataframe
            dataframe containing the images. It is used only if no azimuth_masks are given.
        azimuth_masks : Pandas dataframe
            dataframe containing the masks created with the function azimuth_bias_mask. All masks 
            with an azimuth value between az_min and az_max are plotted. If the value is None,
            the masks are created with the ccditems dataframe.
        az_list : list of float
            list of azimuth value. The regression is made on azimuth angle intervalls with the
            given angles as center points
        sampling_rate : timedelta
            sampling rate of the NADIR images. Default value is def_sampling_rate
        pix_shift : float
            pixel shift along the y axis between 2 consecutive images. For a sampling time of 2s it is 
            equivalent to 2.6 pixels. Default value is def_pix_shift
        
    Returns:
        None
    """
    
    if type(azimuth_masks) != type(None):
        for j in range(len(azimuth_masks)):
            if az_min < azimuth_masks.iloc[j]['azimuth'] and azimuth_masks.iloc[j]['azimuth'] < az_max:
                bias_mask = azimuth_masks['bias_mask'].iloc[j]
                R2_mask = azimuth_masks['R2_mask'].iloc[j]
                az = azimuth_masks['azimuth'].iloc[j]

                fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
                fig.suptitle(f" solar azimuth angle : {az:.2f} deg (extracted from computed masks)")
                fig = ax1.imshow(bias_mask,origin='lower')
                ax1.set_title('bias mask')
                plt.colorbar(fig,ax=ax1,fraction=0.02)
                    
                fig = ax2.imshow(R2_mask,vmin=0.5,vmax=1.0,origin='lower')
                plt.colorbar(fig,ax=ax2,fraction=0.02)
                ax2.set_title('R2 values')
                plt.show()

                plt.figure()
                plt.title(f'Bias histogram (solar azimuth angle : {az:.2f} deg)')
                plt.hist(bias_mask.ravel(),30)
                plt.xlabel('Bias value')
                plt.ylabel('Number of pixels')
                plt.show()


    else : 

        n= len(ccditems)

        if ccditems.iloc[0]['DataLevel'] == 'L1B' :
            im_key = 'ImageCalibrated'
            print('DataLevel L1b')
        elif ccditems.iloc[0]['DataLevel'] == 'L1A' :
            im_key = 'IMAGE'
            print('DataLevel L1a')
        else :
            warnings.warn('DataLevel not recognized (should be L1A or L1B)')

        a,b = np.shape(ccditems.iloc[0][im_key])
        im_points = np.zeros((n,a,b)) # array containing all the pixel values
        NADIR_AZ = np.array(ccditems['nadir_az']) # list of nadir solar azimuth angles
        EXP_DATE = np.array(ccditems['EXPDate']) # list of exposition dates
        
        IM_BIAS = [] # list of bias masks
        IM_R2 = []  # list of R2 masks
    
        # filling the several arrays
        for i in range(n):
            ccditem = ccditems.iloc[i]        
            im = ccditem[im_key]
            im_points [i,:,:] = im       
        
        # intermediate function for regression (only the bias is optimized)
        def func(X,b):
            return X+b
        
        im_shift, pix_ref = nadir_shift(im_ref_def,sampling_rate,pix_shift)
                    
        # regression on the selected images
        im_R2 = np.ones_like(im_points[0,:,:]) # R2 values for each pixel
        im_bias = np.zeros_like(im_points[0,:,:]) # bias value for each pixel

        
        for x_cor in tqdm(range(b),desc='bias analysis'):
            for y_cor in range(a):
                
                step = im_shift[y_cor,x_cor] # images offset between the reference and corrected pixel
                y_ref = pix_ref[y_cor,x_cor,0] # column index of the pixel taken as reference
                x_ref = pix_ref[y_cor,x_cor,1] # row index of the pixel taken as reference

                if not np.isnan(step): # if the pixel is not a reference pixel 
                    step = int(step)
                    y_ref = int(y_ref)
                    x_ref = int(x_ref)
                    #print(x_cor,y_cor,step,y_ref)

                    X = [] # list of reference pixel values
                    Y = [] # list of artifact pixel values
                    REG_AZ = [] # list of azimuth angles of the images

                    az_index_cor = (az_min < NADIR_AZ) & (NADIR_AZ < az_max) # indices for the images with in the given azimuth angle interval (the pixels to be corrected are taken from these images)
                    if step > 0:
                        cor_indexes = np.arange(n)[az_index_cor][:-step] # indices of the corresponding reference images where the reference pixel is taken
                    else:
                        cor_indexes = np.arange(n)[az_index_cor][step:] # indices of the corresponding reference images where the reference pixel is taken


                    if len(cor_indexes) > 0 : 
                        for art_ind in cor_indexes: # iterating over the images (image index of the artifact pixel)
                            ref_ind = art_ind + step # image index of the reference 
                            #check if the image taken as reference is indeed taken the right amount of time after the other image
                            if (EXP_DATE[art_ind] - EXP_DATE[ref_ind] - step*sampling_rate) < timedelta(seconds= 0.01):
                                X.append(im_points[ref_ind,y_ref,x_ref])
                                Y.append(im_points[art_ind,y_cor,x_cor])
                                REG_AZ.append(NADIR_AZ[art_ind])

                    if len(X) + len(Y) > 0:
                    # linear regression, the slope is set to 1 
                        fit_param, cov = curve_fit(func,X,Y)
                        abs_err = Y-func(X,fit_param[0])
                        rsquare = 1.0 - (np.var(abs_err)/np.var(Y))
                        intercept = fit_param[0]
                        im_R2[y_cor,x_cor] = rsquare
                        im_bias[y_cor,x_cor] = intercept

                    else : 
                        im_R2[y_cor,x_cor] = None
                        im_bias[y_cor,x_cor] = None                                 
            
                
        fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
        fig.suptitle(f"{az_min} deg < solar azimuth angle < {az_max} deg")
        fig = ax1.imshow(im_bias,origin='lower')
        ax1.set_title('bias mask')
        plt.colorbar(fig,ax=ax1,fraction=0.02)
            
        fig = ax2.imshow(im_R2,vmin=0.5,vmax=1.0,origin='lower')
        plt.colorbar(fig,ax=ax2,fraction=0.02)
        ax2.set_title('R2 values')
        plt.show()

        plt.figure()
        plt.title(f'Bias histogram ; {az_min} deg < solar azimuth angle < {az_max} deg')
        plt.hist(im_bias.ravel(),30)
        plt.xlabel('Bias value')
        plt.ylabel('Number of pixels') 
        plt.show()




                        
    
    





#%%
df1a = read_MATS_data(datetime(2023,4,4,0,13),datetime(2023,4,14,0,4),level='1a',version='0.6',filter={'CCDSEL':7})

df1a = read_MATS_data(datetime(2023,3,1),datetime(2023,3,15),level='1a',version='0.6',filter={'CCDSEL':7})

azimuth_masks = azimuth_bias_mask(df1a,bias_threshold=0.1,az_list=None,sampling_rate=def_sampling_rate,pix_shift=def_pix_shift)


# %%
reg_analysis(df1a,15,10,-91,-90)

#%%
bias_analysis_angle(15,10,azimuth_masks=azimuth_masks)

#%%
bias_analysis_angle(15,10,ccditems=df1a,az_list=np.linspace(-100,-80,30))
# %%
bias_analysis_histo(-91,-90,azimuth_masks=azimuth_masks)

# %%
bias_analysis_histo(-91,-90,ccditems=df1a)




#%% # run calibration on L1b data 
# the artifact correction is the last correction to be applied during the l1a to l1b processing. The data used to determine the different masks is L1b data processed locally
# with a modified version of the calibration_data.toml file applying blank artifact calibration masks. 
#

from mats_l1_processing.instrument import Instrument
from mats_l1_processing.L1_calibrate import L1_calibrate
from mats_l1_processing.read_parquet_functions import dataframe_to_ccd_items


import os

os.chdir('/home/louis/MATS')
calibration_file='/home/louis/MATS/calibration_data/calibration_data_artifact_analysis.toml' # modified calibration file without artifact correction
instrument=Instrument(calibration_file)

#%%
import time


from mats_l1_processing.L1_calibrate import calibrate_all_items

CCDitems = df1a

CCDitems = dataframe_to_ccd_items(CCDitems)

calibrate_all_items(CCDitems,instrument)
for CCDitem in CCDitems:
    CCDitem['DataLevel']='L1B'

df1b_no_art_correction = pd.DataFrame(CCDitems)
df1b_no_art_correction.rename(columns={'EXP Date':'EXPDate'},inplace=True)

df1b_no_art_correction.to_pickle('df1b_no_art_correction.pkl')


#%%
# CCDitems = df1a

# CCDitems = dataframe_to_ccd_items(CCDitems)

# outputs = np.array(L1_calibrate(CCDitems[0], instrument))


# for CCDitem in tqdm(CCDitems,desc='L1a->L1b processing'):
#         outputs += np.array(L1_calibrate(CCDitem, instrument))
      
# for i in range(8):
#     plt.figure()
#     plt.imshow(outputs[i])
#     plt.show()

# %%

# azimuth_masks_v2 = azimuth_bias_mask(df1b_no_art_correction,bias_threshold=-56780,az_list=None,sampling_rate=def_sampling_rate,pix_shift=def_pix_shift)

# # save_masks(azimuth_masks_v2,'azimuth_masks_v2.pkl')

# azimuth_masks_v2.to_pickle('azimuth_masks_v2.pkl')

# # %%
# reg_analysis(df1b_no_art_correction,15,10,-91,-90)
# bias_analysis_angle(15,10,azimuth_masks=azimuth_masks_v2)
# bias_analysis_angle(15,10,ccditems=df1b_no_art_correction,az_list=np.linspace(-100,-80,30))
# bias_analysis_histo(-91,-90,azimuth_masks=azimuth_masks_v2)
# bias_analysis_histo(-91,-90,ccditems=df1b_no_art_correction)
# %%
from mats_l1_processing.L1_calibrate import calibrate_all_items

start_time = datetime(2023,4,13)
end_time = datetime(2023,4,14)

while end_time < datetime(2023,6,1):
    print(start_time,end_time)
    try :
        df1a = read_MATS_data(start_time,end_time,level='1a',version='0.6',filter={'CCDSEL':7})
        # azimuth_masks_l1a = azimuth_bias_mask(df1a,bias_threshold=-5445754547,az_list=None,sampling_rate=def_sampling_rate,pix_shift=def_pix_shift)
        # azimuth_masks_l1a.to_pickle(f'azimuth_masks_l1a_{start_time.strftime("%Y%m%d")}.pkl')
        CCDitems = df1a
        CCDitems = dataframe_to_ccd_items(CCDitems)
        calibrate_all_items(CCDitems,instrument)
        for CCDitem in CCDitems:
            CCDitem['DataLevel']='L1B'
        df1b_no_art_correction = pd.DataFrame(CCDitems)
        df1b_no_art_correction.rename(columns={'EXP Date':'EXPDate'},inplace=True)
        df1b_no_art_correction.to_pickle(f'df1b_no_art_correction_{start_time.strftime("%Y%m%d")}.pkl')
        # azimuth_masks_l1b = azimuth_bias_mask(df1b_no_art_correction,bias_threshold=-56780,az_list=None,sampling_rate=def_sampling_rate,pix_shift=def_pix_shift)
        # azimuth_masks_l1b.to_pickle(f'azimuth_masks_l1b_{start_time.strftime("%Y%m%d")}.pkl')
    except :
        print('no data')
    start_time = end_time
    end_time += timedelta(days=1)
# %%

os.chdir('/home/louis/MATS')

dir_calib_list = ['df1b_no_art_correction_20230301.pkl',
                  'df1b_no_art_correction_20230308.pkl',
                  'df1b_no_art_correction_20230315.pkl',
                  'df1b_no_art_correction_20230322.pkl',
                  'df1b_no_art_correction_20230329.pkl',
                  'df1b_no_art_correction_20230405.pkl',
                  'df1b_no_art_correction_20230412.pkl',
                  'df1b_no_art_correction_20230419.pkl',
                  'df1b_no_art_correction_20230426.pkl',
                  'df1b_no_art_correction_20230503.pkl',
                  'df1b_no_art_correction_20230511.pkl']

import pickle

with open(dir_calib_list[0], 'rb') as f:
    df1b_no_art_correction = pickle.load(f)

for dir_calib in dir_calib_list[1:]:
    with open(dir_calib, 'rb') as f:
        df1b_no_art_correction = pd.concat([df1b_no_art_correction,pickle.load(f)])


reg_analysis(df1b_no_art_correction,15,10,-91,-90)
#bias_analysis_angle(15,10,azimuth_masks=azimuth_masks_v2)
bias_analysis_angle(15,10,ccditems=df1b_no_art_correction,az_list=np.linspace(-100,-80,30))
#bias_analysis_histo(-91,-90,azimuth_masks=azimuth_masks_v2)
bias_analysis_histo(-91,-90,ccditems=df1b_no_art_correction)

azimuth_masks_l1b = azimuth_bias_mask(df1b_no_art_correction,bias_threshold=-56780,az_list=None,sampling_rate=def_sampling_rate,pix_shift=def_pix_shift)


# %%
