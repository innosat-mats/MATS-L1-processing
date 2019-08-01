#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:09:57 2019

@author: franzk

functions used for MATS L1 processing, based on corresponding MATLAB scripts provided by
Georgi Olentsenko and Mykola Ivchenko
The MATLAB script can be found here: https://github.com/OleMartinChristensen/MATS-image-analysis
"""

import numpy as np
import time
import matplotlib.pyplot as plt

#ismember function taken from https://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function
#needs to be tested, if actually replicates matlab ismember function

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value



def readimg(filename):

    data_arr=np.fromfile(filename, dtype='uint16')#check endianess, uint16 instead of ubit16 seems to work
    #convert header to binary
    #header_bin = [bin(np.double(data_arr[i])) for i in range(0,12)]#check if at least 16 bits, check if indexing correct
    header_bin = np.asarray([bin(data_arr[i]) for i in range(0,12)])#this is a string 
    #change format of header_bin elements to be formatted like matlab char array
    for i in range(0,len(header_bin)):
        header_bin[i]=header_bin[i][2:].zfill(16)
    #read header
    Frame_count = int(header_bin[0],2)
    NRow = int(header_bin[1],2)
    NRowBinCCD = int(header_bin[2][10:16],2)
    NRowSkip = int(header_bin[3][8:16],2)
    NCol = int(header_bin[4],2)
    NColBinFPGA = int(header_bin[5][2:8],2)
    NColBinCCD = int(header_bin[5][8:16],2)
    NColSkip = int(header_bin[6][5:16],2)
    N_flush = int(header_bin[7],2)
    Texposure_MSB = int(header_bin[8],2)
    Texposure_LSB = int(header_bin[9],2)
    Gain = int(header_bin[10],2)
    SignalMode = Gain & 4096
    Temperature_read = int(header_bin[11],2)

    #read image
    if len(data_arr)< NRow*(NCol+1)/(2**(NColBinFPGA)):#check for differences in python 2 and 3
        img_flag = 0
        image = 0
        Noverflow = 0
        BlankLeadingValue = 0
        BlankTrailingValue = 0
        ZeroLevel = 0
        
        Reserved1 = 0
        Reserved2 = 0
        
        Version = 0
        VersionDate = 0
        NBadCol = 0
        BadCol = 0
        Ending = 'Wrong size'
    else:
        img_flag = 1
        image = np.reshape(np.double(data_arr[11+1:NRow*(NCol+1)+12]), (NCol+1,NRow))
        image = np.matrix(image).getH()
    
        #Trailer
        trailer_bin = np.asarray([bin(data_arr[i]) for i in range(NRow*(NCol+1)+12,len(data_arr))])
        for i in range(0,len(trailer_bin)):
            trailer_bin[i]=trailer_bin[i][2:].zfill(16)
        Noverflow = int(trailer_bin[0],2)
        BlankLeadingValue = int(trailer_bin[1],2)
        BlankTrailingValue = int(trailer_bin[2],2)
        ZeroLevel = int(trailer_bin[3],2)
        
        Reserved1 = int(trailer_bin[4],2)
        Reserved2 = int(trailer_bin[5],2)
        
        Version = int(trailer_bin[6],2)
        VersionDate = int(trailer_bin[7],2)
        NBadCol = int(trailer_bin[8],2)
        BadCol = []
        Ending = int(trailer_bin[-1],2)
        
        if NBadCol > 0:
            BadCol = np.zeros(NBadCol)
            for k_bc in range(0,NBadCol):
                BadCol[k_bc] = int(trailer_bin[9+k_bc],2)#check if this actually works

    #original matlab code uses structured array, as of 20-03-2019 implementation as dictionary seems to be more useful choice
    #decision might depend on further (computational) use of data, which is so far unknown to me
    header = {}
    header['Size'] = len(data_arr)
    header['Frame_count'] = Frame_count
    header['NRow'] = NRow
    header['NRowBinCCD'] = NRowBinCCD
    header['NRowSkip'] = NRowSkip
    header['NCol'] = NCol
    header['NColBinFPGA'] = NColBinFPGA
    header['NColBinCCD'] = NColBinCCD
    header['NColSkip'] = NColSkip
    header['N_flush'] = N_flush
    header['Texposure'] = Texposure_LSB + Texposure_MSB*2**16
    header['Gain'] = Gain & 255
    header['SignalMode'] = SignalMode
    header['Temperature_read'] = Temperature_read
    header['Noverflow'] = Noverflow
    header['BlankLeadingValue'] = BlankLeadingValue
    header['BlankTrailingValue'] = BlankTrailingValue
    header['ZeroLevel'] = ZeroLevel
    header['Reserved1'] = Reserved1
    header['Reserved2'] = Reserved2
    header['Version'] = Version
    header['VersionDate'] = VersionDate
    header['NBadCol'] = NBadCol
    header['BadCol'] = BadCol
    header['Ending'] = Ending        
    
    return image, header, img_flag


def readimgpath(path, file_number, plot):
    #stuff happens
    filename = '%sF_0%02d/D_0%04d' % (path, np.floor(file_number/100),file_number)
    image, header, img_flag = readimg(filename)

    if plot>0:
        #do the plotting
        mean_img=np.mean(image)
    
        plt.imshow(image, cmap='viridis', vmin=mean_img-100, vmax=mean_img+100)
        plt.title('CCD image')
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        plt.show()#equivalent to hold off
        
        print(header)
        time.sleep(0.1)
    
    return image, header


def predict_image(reference_image, hsm_header, lsm_image, lsm_header, header):
    """
    this is a function to predict an image read out from the CCD with a given set
    of parameters, based on a reference image (of size 511x2048)
    """
    ncol = int(header['NCol']) + 1
    nrow = int(header['NRow'])
    
    nrowskip = int(header['NRowSkip'])
    ncolskip = int(header['NColSkip'])
    
    nrowbin = int(header['NRowBinCCD'])
    ncolbinC = int(header['NColBinCCD'])
    ncolbinF = 2**int(header['NColBinFPGA'])
    
    if int(header['SignalMode']) > 0:
        blank=int(lsm_header['BlankTrailingValue'])
    else:
        blank=int(hsm_header['BlankTrailingValue'])
        
    blank_off=blank-128
    zerolevel=int(header['ZeroLevel'])
    
    gain=2**(int(header['Gain']) & 255)
    
    bad_columns = header['BadCol']
    
    if nrowbin == 0:# no binning means beaning of one
        nrowbin=1
    
    if ncolbinC == 0:# no binning means beaning of one
        ncolbinC=1
    
    if ncolbinF == 0:# no binning means beaning of one
        ncolbinF=1
        
    ncolbintotal = ncolbinC*ncolbinF
    
    if int(header['SignalMode']) > 0:
        reference_image = get_true_image(lsm_image, lsm_header)
        reference_image = desmear_true_image(reference_image, lsm_header)
    else:
        reference_image = get_true_image(reference_image, hsm_header)
        reference_image = desmear_true_image(reference_image, hsm_header)
    
    #bad column analysis
    n_read, n_coadd = binning_bc(ncol, ncolskip, ncolbinF, ncolbinC, header['BadCol'])
    
    image=np.zeros((nrow,ncol))
    image[:,:]=128 #offset
    
    finished_row = 0
    finished_col = 0
    for j_r in range(0,nrow):#check indexing again
        for j_c in range(0,ncol):
            for j_br in range(0,nrowbin): #account for row binning on CCD
                if j_br==0:
                    image[j_r, j_c] = image[j_r, j_c] + n_read[j_c]*blank_off # here we add the blank value, only once per binned row
                for j_bc in range(0,ncolbintotal): # account for column binning
                    #out of reference image range
                    if (j_r)*nrowbin + j_br + nrowskip > 511:
                        break
                    elif (j_c)*ncolbinC*ncolbinF + j_bc + ncolskip > 2048:
                        break
                    
                    #removed +1 after bad_columns, unclear why it was added
                    #TODO
                    if ncolbinC > 1 and (j_c)*ncolbinC*ncolbinF + j_bc + ncolskip in bad_columns:# +1 becuase Ncol is +1
                        continue
                    else:
                        #add only the actual signal from every pixel (minus blank)
                        image[j_r, j_c] = (image[j_r, j_c] #remove blank
                        + reference_image[(j_r-1)*nrowbin+j_br+nrowskip-1,(j_c-1)*ncolbinC*ncolbinF+j_bc+ncolskip-1] #row and column value evaluation, -1 to adjust for python indexing
                        * 1 #scaling factor
                        )

    image = image/gain
    pred_header = header
    pred_header['BlankTrailingValue'] = blank
    
    return image, pred_header

def get_true_image(image, header):
    #calculate true image by removing readout offset, pixel blank value and
    #normalising the signal level according to readout time
    
    ncolbinC=int(header['NColBinCCD'])
    if ncolbinC == 0:
        ncolbinC = 1
    
    #remove gain
    true_image = image * 2**(int(header['Gain']) & 255)
    
    #bad column analysis
    n_read, n_coadd = binning_bc(int(header['NCol'])+1, int(header['NColSkip']), 2**int(header['NColBinFPGA']), ncolbinC, header['BadCol'])
    
    #go through the columns
    for j_c in range(0, int(header['NCol'])):
        #remove blank values and readout offsets
        true_image[0:int(header['NRow']), j_c] = true_image[0:int(header['NRow']), j_c] - n_read[j_c]*(header['BlankTrailingValue']-128)-128
        
        #compensate for bad columns
        true_image[0:int(header['NRow']), j_c] = true_image[0:int(header['NRow']), j_c] * (2**int(header['NColBinFPGA'])*ncolbinC/n_coadd[j_c])
    
    return true_image

def binning_bc(Ncol, Ncolskip, NcolbinFPGA, NcolbinCCD, BadColumns):
    
    #a routine to estimate the correction factors for column binning with bad columns
    
    #n_read - array, containing the number of individually read superpixels
    #           attributing to the given superpixel
    #n_coadd - array, containing the number of co-added individual pixels
    #Input - as per ICD. BadColumns - array containing the index of bad columns
    #           (the index of first column is 0)
    
    n_read=np.zeros(Ncol)
    n_coadd=np.zeros(Ncol)
    
    col_index=Ncolskip
    
    for j_col in range(0,Ncol):
        for j_FPGA in range(0,NcolbinFPGA):
            continuous=0
            for j_CCD in range(0,NcolbinCCD):
                if col_index in BadColumns:
                    if continuous == 1:
                        n_read[j_col]=n_read[j_col]+1
                    continuous=0
                else:
                    continuous=1
                    n_coadd[j_col]=n_coadd[j_col]+1
                
                col_index=col_index+1
            
            if continuous == 1:
                n_read[j_col]=n_read[j_col]+1
    
    return n_read, n_coadd


def desmear_true_image(image, header):
    
    nrow = int(header['NRow'])
    ncol = int(header['NCol']) + 1
    
    #calculate extra time per row
    T_row_extra, T_delay = calculate_time_per_row(header)
    T_exposure = int(header['Texposure'])/1000#check for results when shifting from python 2 to 3
    
    for irow in range(1,nrow):
        for krow in range(0,irow):
            image[irow,0:ncol]=image[irow,0:ncol] - image[krow,0:ncol]*(T_row_extra/T_exposure)
            
    #row 0 here is the first row to read out from the chip
    
    return image

def calculate_time_per_row(header):
    
    #this function provides some useful timing data for the CCD readout
    
    #Note that minor "transition" states may have been omitted resulting in 
    #somewhat shorter readout times (<0.1%).
    
    #Default timing setting is_
    #ccd_r_timing <= x"A4030206141D"&x"010303090313"
    
    #All pixel timing setting is the final count of a counter that starts at 0,
    #so the number of clock cycles exceeds the setting by 1
    
    #image parameters
    ncol=int(header['NCol'])+1
    ncolbinC=int(header['NColBinCCD'])
    if ncolbinC == 0:
        ncolbinC = 1
    ncolbinF=2**int(header['NColBinFPGA'])
    
    nrow=int(header['NRow'])
    nrowbin=int(header['NRowBinCCD'])
    if nrowbin == 0:
        nrowbin = 1
    nrowskip=int(header['NRowSkip'])
    
    n_flush=int(header['N_flush'])
    
    #timing settings
    full_timing = 1 #TODO <-- meaning?
    
    #full pixel readout timing
    time0 = 1 + 19 # x13%TODO
    time1 = 1 +  3 # x03%TODO
    time2 = 1 +  9 # x09%TODO
    time3 = 1 +  3 # x03%TODO
    time4 = 1 +  3 # x03%TODO
    time_ovl = 1 + 1 # x01%TODO
    
    # fast pixel readout timing
    timefast  = 1 + 2 # x02%TODO
    timefastr = 1 + 3 # x03%TODO
    
    #row shift timing
    row_step = 1 + 164 # xA4%TODO
    
    clock_period = 30.517 #master clock period, ns 32.768 MHz
    
    #there is one extra clock cycle, effectively adding to time 0
    Time_pixel_full = (1+ time0 + time1 + time2 + time3 + time4 + 3*time_ovl)*clock_period
    
    # this is the fast timing pixel period
    Time_pixel_fast = (1+ 4*timefast + 3*time_ovl + timefastr)*clock_period
    
    #here we calculate the number of fast and slow pixels
    #NOTE: the effect of bad pixels is disregarded here
    
    if full_timing == 1:
        n_pixels_full = 2148
        n_pixels_fast = 0
    else:
        if ncolbinC < 2: #no CCD binning
            n_pixels_full = ncol * ncolbinF
        else: #there are two "slow" pixels for one superpixel to be read out
            n_pixels_full = 2*ncol *ncolbinF
        n_pixels_fast = 2148 - n_pixels_full
    
    
    #time to read out one row
    T_row_read = n_pixels_full*Time_pixel_full + n_pixels_fast*Time_pixel_fast
    
    # shift time of a single row
    T_row_shift = (64 + row_step *10)*clock_period
    
    #time of the exposure start delay from the start_exp signal # n_flush=1023
    T_delay = T_row_shift * n_flush
    
    #total time of the readout
    T_readout = T_row_read*(nrow+nrowskip+1) + T_row_shift*(1+nrowbin*nrow)
    
    
    #"smearing time"
    #(this is the time that any pixel collects electrons in a wrong row, during the shifting.)
    #For smearing correction, this is the "extra exposure time" for each of the rows.
    
    T_row_extra = (T_row_read + T_row_shift*nrowbin) / 1e9    
    
    return T_row_extra, T_delay

def compare_image(image1, image2, header):
    
    #this is a function to compare two images of the same size
    #one comparison is a linear fit of columns, the other comparison is a linear fit
    #of rows, the third is a linear fit of the whole image
    
    sz1=image1.shape
    sz2=image2.shape
    
    if sz1[0] != sz2[0] or sz1[1] != sz2[1]:
        print('sizes of input images do not match')
    
    nrow=sz1[0]
    ncol=sz1[1]
    
    nrowskip = int(header['NRowSkip'])
    ncolskip = int(header['NColSkip'])
    
    nrowbin = int(header['NRowBinCCD'])
    ncolbinC = int(header['NColBinCCD'])
    ncolbinF = 2**int(header['NColBinFPGA'])
    
    if nrowskip + nrowbin*nrow > 511:
        nrow = int(np.floor((511-nrowskip)/nrowbin))
        
    if ncolskip + ncolbinC*ncolbinF*ncol > 2047:
        ncol = int(np.floor((2047-ncolskip)/(ncolbinC*ncolbinF)))#calculation for nrow in original, seems questionable

    image1 = image1[0:nrow, 0:ncol]
    image2 = image2[0:nrow, 0:ncol]

    r_scl=np.zeros(nrow)
    r_off=np.zeros(nrow)
    r_std=np.zeros(nrow)

    for jj in range(0,nrow):
        
        x=np.concatenate((np.ones((ncol,1)), np.expand_dims(image1[jj,].conj().transpose(), axis=1)), axis=1)#-1 to adjust to python indexing?
        y=image2[jj,].conj().transpose()
        bb, ab, aa, cc = np.linalg.lstsq(x,y)
        
        ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
        #ft=np.multiply(x[:,1]*bb[1]) + bb[0]

        adf=np.abs(np.squeeze(y)-np.squeeze(ft))
        sigma=np.std(np.squeeze(y)-np.squeeze(ft))

        inside = np.where(adf < 2*sigma)
        bb, ab, aa, cc = np.linalg.lstsq(x[inside[1],], y[inside[1]])
        
        ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
        
        r_scl[jj]=bb[1]
        r_off[jj]=bb[0]
        r_std[jj]=np.std(y[0]-ft[0])
        
    c_scl=np.zeros(ncol)
    c_off=np.zeros(ncol)
    c_std=np.zeros(ncol)
   
    for jj in range(0,ncol):
        
        x=np.concatenate((np.ones((nrow,1)), np.expand_dims(image1[:,jj], axis=1)), axis=1)
        y=image2[:,jj]
        bb, ab, aa, cc = np.linalg.lstsq(x,y)
        
        ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
        
        adf=np.abs(np.squeeze(y)-np.squeeze(ft))
        sigma=np.std(np.squeeze(y)-np.squeeze(ft))

        inside = np.where(adf < 2*sigma)
        bb, ab, aa, cc = np.linalg.lstsq(x[inside[1],], y[inside[1]])
            
        ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
        c_scl[jj]=bb[1]
        c_off[jj]=bb[0]
        c_std[jj]=np.std(y[0]-ft[0])
    
    nsz=(nrow)*(ncol)
    la_1=np.reshape(image1, (nsz,1))
    la_2=np.reshape(image2, (nsz,1))
    
    x=np.concatenate((np.ones((nsz,1)),la_1), axis=1)
    y=la_2
    bb, ab, aa, cc = np.linalg.lstsq(x,y)
    
    ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
    
    adf=np.abs(np.squeeze(y)-np.squeeze(ft))
    sigma=np.std(np.squeeze(y)-np.squeeze(ft))
    
    inside = np.where(adf < 2*sigma)
    bb, ab, aa, cc = np.linalg.lstsq(x[inside[1],], y[inside[1]])
    
        
    ft=np.squeeze([a*bb[1] for a in x[:,1]]) + bb[0]
    
    t_off=bb[0]
    t_scl=bb[1]
    t_std=np.std(y[0]-ft[0])
    
    rows=0
    
    return t_off, t_scl, t_std

def compensate_bad_columns(image, header):
    
    #this is a function to compensate bad columns if in the image
    
    ncol=int(header['NCol'])+1
    nrow=int(header['NRow'])
    
    ncolskip=int(header['NColSkip'])
    
    ncolbinC=int(header['NColBinCCD'])
    ncolbinF=2**int(header['NColBinFPGA'])
    
    #change to Leading if Trailing does not work properly
    blank=int(header['BlankTrailingValue'])
    
    gain=2**(int(header['Gain']) & 255)
    
    if ncolbinC == 0: #no binning means binning of one
        ncolbinC=1
    
    if ncolbinF==0: #no binning means binning of one
        ncolbinF=1
    
    #bad column analysis
    
    n_read, n_coadd = binning_bc(ncol, ncolskip, ncolbinF, ncolbinC, np.asarray(header['BadCol']))
    
    if ncolbinC>1:
        for j_c in range(0,ncol):
            if ncolbinC*ncolbinF != n_coadd[j_c]:
                #remove gain adjustment
                image[0:nrow-1,j_c] = image[0:nrow-1,j_c]*gain
                
                #remove added superpixel value due to bad columns and read out offset
                image[0:nrow-1,j_c] = image[0:nrow-1,j_c] - n_read[j_c]*(blank-128) -128
                
                #multiply by number of binned column to actual number readout ratio
                image[0:nrow-1,j_c] = image[0:nrow-1,j_c] * ((ncolbinC*ncolbinF)/n_coadd[j_c])
                
                #add number of FPGA binned
                image[0:nrow-1,j_c] = image[0:nrow-1,j_c] + ncolbinF*(blank-128) + 128
                
                #add gain adjustment back
                image[0:nrow-1,j_c] = image[0:nrow-1,j_c]/gain
                
                print('Col: ',j_c,', n_read: ',n_read[j_c],', n_coadd: ',n_coadd[j_c],', binned pixels: ',ncolbinC*ncolbinF)
    
    return image


def get_true_image_from_compensated(image, header):
    
    #calculate true image by removing readout offset, pixel blank value and
    #normalising the signal level according to readout time
    
    #remove gain
    true_image = image * 2**(int(header['Gain']) & 255)
    
    for j_c in range(0,int(header['NCol'])):
        true_image[0:header['NRow'], j_c] = ( true_image[0:header['NRow'],j_c] - 
                  2**header['NColBinFPGA'] * (header['BlankTrailingValue']-128) - 128 )
    
    
    return true_image