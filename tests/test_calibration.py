import pytest

from mats_l1_processing.read_and_calibrate_all_files_parallel import main
from mats_l1_processing.instrument import Instrument, CCD

import pickle
import numpy as np
import matplotlib.pyplot as plt


__author__ = "Ole Martin Christensen"
__copyright__ = "Ole Martin Christensen"
__license__ = "MIT"


def test_calibrate():
    main("testdata/RacFiles_out/", "tests/calibration_data_test.toml")


# def test_plot():
#     main(
#         "testdata/RacFiles_out/",
#         "tests/calibration_data_test.toml",
#         calibrate=False,
#         plot=True,
#     )

def test_readfunctions():
    from mats_l1_processing.read_in_functions import readprotocol, read_all_files_in_root_directory
    from mats_l1_processing.LindasCalibrationFunctions import read_all_files_in_protocol
    

    directory='testdata/210215OHBLimbImage/'
    protocol='protocol_dark_bright_100um_incl_IR3.txt'


    read_from="rac" 
    df_protocol=readprotocol(directory+protocol)

    df_bright=df_protocol[df_protocol.DarkBright=='B']
    CCDitems=read_all_files_in_protocol(df_bright, read_from,directory)

    with open('testdata/CCDitem_example.pkl', 'wb') as f:
        pickle.dump(CCDitems[0], f)

    CCDitems=read_all_files_in_root_directory(read_from,directory)
    
    read_from="imgview" 
    CCDitems=read_all_files_in_root_directory(read_from,directory)
    

    
def test_CCDunit():
    intrument = Instrument("tests/calibration_data_test.toml")
    CCDunit_IR1=intrument.get_CCD("IR1")
    with open('testdata/CCDunit_IR1_example.pkl', 'wb') as f:
        pickle.dump(CCDunit_IR1, f)

def test_forward_backward(): 
    """
    This tests the forward and backward calibration. 
    The backward calibraton should completely reverse everything the forward 
    calibration has done thus giving back the original image.
    This test needs a CCDitem and a CCDunit, which are created and saved in 
    test_reafunctions and test_CCDunit.
    
    """
    from mats_l1_processing.forward_model import  forward_and_backward
    
    with open('testdata/CCDitem_example.pkl', 'rb') as f:
        CCDitem = pickle.load(f)
    
    with open('testdata/CCDunit_IR1_example.pkl', 'rb') as f:
        CCDunit_IR1=pickle.load(f)        
    CCDitem['CCDunit']=CCDunit_IR1

    forward_and_backward(CCDitem,  photons=1000, plot=False)

if __name__ == "__main__":

    test_readfunctions()
    test_CCDunit()
    test_forward_backward()