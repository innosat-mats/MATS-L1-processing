#%%
import json
from re import A
from xml.etree.ElementPath import xpath_tokenizer_re
import numpy as np
from matplotlib import pyplot as plt
import pickle
import toml
import scipy.optimize as opt

#%%

def generate_CCDItem():

    nrow = 511
    ncol = 2048
    CCDItem= {'File': 'Simulation',
    'ProcessingDate': '2021-01-01T00:00:00+00:00',
    'RamsesTime': '2021-01-01T00:00:00.000Z',
    'QualityIndicator': 0,
    'LossFlag': 0,
    'VCFrameCounter': 0,
    'SPSequenceCount': 0,
    'TMHeaderTime': '2021-01-01T00:00:00.000000000Z',
    'TMHeaderNanoseconds': 0000000000000,
    'SID': '', 
    'RID': 'CCD1', 
    'CCDSEL': 1, 
    'EXP Nanoseconds': 0000000000000, 
    'EXP Date': '2021-01-01T00:00:00.000000000Z', 
    'WDW Mode': 'Manual', 
    'WDW InputDataWindow': '15..0', 
    'WDWOV': 0, 
    'JPEGQ': 101, 
    'FRAME': 1, 
    'NROW': nrow, 
    'NRBIN': 1, 
    'NRSKIP': 0, 
    'NCOL': ncol-1, 
    'NCBIN FPGAColumns': 1, 
    'NCBIN CCDColumns': 1, 
    'NCSKIP': 0, 
    'NFLUSH': 1023, 
    'TEXPMS': 6000, 
    'GAIN Mode': 'High', 
    'GAIN Timing': 'Faster', 
    'GAIN Truncation': 0, 
    'TEMP': 4307, 
    'FBINOV': 0, 
    'LBLNK': 307, 
    'TBLNK': 313, 
    'ZERO': 121, 
    'TIMING1': 5149, 
    'TIMING2': 518, 
    'VERSION': 50, 
    'TIMING3': 41987, 
    'NBC': 0, 
    'BC': [], 
    'Image File Name': 'simulation.png', 
    'Error': '', 
    'IMAGE': [], 
    'jsondata': '', 
    'read_from': 'rac', 
    'reltime': '', 
    'id': '0000000000000_1', 
    'NColBinCCD': 1, 
    'NColBinFPGA': 0.0, 
    'DigGain': 0, 
    'SigMode': 0, 
    'channel': 'IR1', 
    'OBCtemperature': 20.691176470588232, 
    'temperature': 20.91508892713712, 
    'temperature_HTR': 20.91508892713712, 
    'temperature_ADC': 20.691176470588232}
    
    return CCDItem

# %%
def make_and_save_CCDitem(filename):
    CCDitem = generate_CCDItem()
    
    with open(filename, 'w') as fp:
        json.dump(CCDitem,fp,allow_nan=True)
# %%
