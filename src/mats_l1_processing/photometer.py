#%%
import numpy as np
from mats_l1_processing.instrument import Photometer
from pandas import DataFrame



def calibrate_pm(df: DataFrame, photometer: Photometer):
    
# ===============================================================================================================
# Calibration starts here
# PM1 is Bkg photometer, 754BP3 nm / PM2 is A-band photometer, 763BP8 nm
#   Change to bits/s

    SAMPLING_CYCLES = 166

    PM1_Sig_bit = df['PM1S']/df['PM1SCNTR']
    PM1_Tpd_bit = df['PM1A']/df['PM1ACNTR']
    PM1_Tif_bit = df['PM1B']/df['PM1BCNTR']
    PM2_Sig_bit = df['PM2S']/df['PM2SCNTR']
    PM2_Tpd_bit = df['PM2A']/df['PM2ACNTR']
    PM2_Tif_bit = df['PM2B']/df['PM2BCNTR']
    pm_texp = round(df['PM1SCNTR']/SAMPLING_CYCLES) # exposure time in seconds, 1 second "integration" is 166-167 sampling cycles

# ===========================================================
# Calibrate data

#   extracting variables from dict
    bitar = photometer.cal_therm['bitar'] # 0.1 to 4095 bits in steps of 0.1 bit, array of float64 (1,40950)
    TempFM1if_raw = photometer.cal_therm['TempFM1if_raw'] # array of float64 (1,40950)
    TempFM1pd_raw = photometer.cal_therm['TempFM1pd_raw'] # array of float64 (1,40950)
    TempFM2if_raw = photometer.cal_therm['TempFM2if_raw'] # array of float64 (1,40950)
    TempFM2pd_raw = photometer.cal_therm['TempFM2pd_raw'] # array of float64 (1,40950)
    SignFM1_Rad_raw = photometer.cal_rad_FM1 # spline representing radiance from bits and temperature
    SignFM2_Rad_raw = photometer.cal_rad_FM2 # spline representing radiance from bits and temperature

# =================================
# Change raw temperature data to °C
#   define pmBkg_Tpd, pmBkg_Tif, pmAband_Tpd, pmAband_Tif
    pmBkg_Tpd = np.NaN * np.ones_like(PM1_Tpd_bit)   # Temperature of the Bkg photometer photodiode
    pmBkg_Tif = np.NaN * np.ones_like(PM1_Tif_bit)   # Temperature of the Bkg photometer filter
    pmAband_Tpd = np.NaN * np.ones_like(PM2_Tpd_bit) # Temperature of the A-band photometer photodiode
    pmAband_Tif = np.NaN * np.ones_like(PM2_Tif_bit) # Temperature of the A-band photometer filter

    for ij in range(0, len(PM1_Tpd_bit)-1):
        index11 =  np.where(bitar == round(PM1_Tpd_bit[ij],1))
        pmBkg_Tpd[ij] = TempFM1pd_raw[index11]
        index12 =  np.where(bitar == round(PM1_Tif_bit[ij],1))
        pmBkg_Tif[ij] = TempFM1if_raw[index12]
        index13 =  np.where(bitar == round(PM2_Tpd_bit[ij],1))
        pmAband_Tpd[ij] = TempFM2pd_raw[index13]
        index14 =  np.where(bitar == round(PM2_Tif_bit[ij],1))
        pmAband_Tif[ij] = TempFM2if_raw[index14]

# ==============================
# Change raw photometer data to photons cm-2 str-1 s-1
#   define pmBkg_Sig, pmAband_Sig
    pmBkg_Sig = np.ones_like(PM1_Sig_bit)   # Background photometer signal
    pmAband_Sig = np.ones_like(PM2_Sig_bit) # A-band photometer signal

    for ik in range(0, len(PM1_Sig_bit)-1):

        pmBkg_Sig[ik] = SignFM1_Rad_raw(pmBkg_Tpd[ik],PM1_Sig_bit[ik])
        pmAband_Sig[ik] = SignFM2_Rad_raw(pmAband_Tpd[ik],PM2_Sig_bit[ik])

        # index21 = np.where(Temperatur == round(pmBkg_Tpd[ik],1)) #OMC 2023.04.04: Shoud interpolation be used insted of lookup table?
        # index22 = np.where(Temperatur == round(pmAband_Tpd[ik],1)) #OMC 2023.04.04: Shoud interpolation be used insted of lookup table?
    
        # index23 = np.where(bitar == round(PM1_Sig_bit[ik],1)) #OMC 2023.04.04: Shoud interpolation be used insted of lookup table?
        # if index23[1].size == 0:
        #     pmBkg_Sig[ik] = np.NaN
        # else:
        #     pmBkg_Sig[ik] = SignFM1_Rad_raw[index21[1], index23[1]]
    
        # index24 = np.where(bitar == round(PM2_Sig_bit[ik],1)) #OMC 2023.04.04: Should interpolation be used insted of lookup table?
        # if index24[1].size == 0:
        #     pmAband_Sig[ik] = np.NaN
        # else:
        #     pmAband_Sig[ik] = SignFM2_Rad_raw[index22[1], index24[1]]

# End of calibration
# ===============================================================================================================
# Modify dataframe before returning it
#   Remove raw photometer data from the df dataframe
    df.drop(df.iloc[:, 12:24], inplace=True, axis=1)

#   Add calibrated data to the df dataframe
    df["pmAband_Sig_bit"] = PM2_Sig_bit # Signal in bits 
    df["pmAband_Sig"] = pmAband_Sig # A-band photometer signal [photons cm-2 str-1 s-1]
    df["pmAband_Tpd"] = pmAband_Tpd # Temperature of the A-band photometer photodiode [°C]
    df["pmAband_Tif"] = pmAband_Tif # Temperature of the A-band photometer interference filter [°C]
    df["pmBkg_Sig_bit"] = PM1_Sig_bit # Signal in bits 
    df["pmBkg_Sig"] = pmBkg_Sig # Background photometer signal [photons cm-2 str-1 s-1]
    df["pmBkg_Tpd"] = pmBkg_Tpd # Temperature of the background photometer photodiode [°C]
    df["pmBkg_Tif"] = pmBkg_Tif # Temperature of the background photometer interference filter [°C]
    df["pm_texp"] = pm_texp # The photometer exposure time [s]

    return(df)
# %%