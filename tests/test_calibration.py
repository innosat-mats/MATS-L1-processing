import pytest

from mats_l1_processing.read_and_calibrate_all_files_parallel import main
from mats_l1_processing.instrument import Instrument, CCD
from mats_l1_processing.L1_calibration_functions import inverse_model_real,inverse_model_table,make_binary,combine_flags

import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

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
    from mats_l1_processing.experimental_utils import read_all_files_in_protocol
    

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

    intrument = Instrument("tests/calibration_data_test.toml")
    CCDunit_IR1=intrument.get_CCD("UV1")
    with open('testdata/CCDunit_UV1_example.pkl', 'wb') as f:
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

def test_non_linearity_fullframe():
    with open('testdata/CCDitem_example.pkl', 'rb') as f:
        CCDitem = pickle.load(f)
    
    with open('testdata/CCDunit_IR1_example.pkl', 'rb') as f:
        CCDunit_IR1=pickle.load(f)        
    CCDitem['CCDunit']=CCDunit_IR1

    table = CCDitem['CCDunit'].get_table(CCDitem)
    ref_table = np.load('testdata/IR1_table.npy')
    assert (table==ref_table).all()

    image_linear_table,error_flag = inverse_model_table(table,0)
    image_linear_real,error_flag = inverse_model_real(CCDitem,0)
    assert image_linear_table==0.0
    assert np.abs(image_linear_real-image_linear_table)<1e-3
    
    image_linear_table,error_flag = inverse_model_table(table,1e3)
    image_linear_real,error_flag = inverse_model_real(CCDitem,1e3)
    assert np.abs(image_linear_real-image_linear_table)<1e-3

    image_linear_table,error_flag = inverse_model_table(table,10e3)
    image_linear_real,error_flag = inverse_model_real(CCDitem,10e3)
    assert np.abs(image_linear_real-image_linear_table)<1e-3

def test_non_linearity_binned():
    with open('testdata/CCDitem_binned_example.pkl', 'rb') as f:
        CCDitem = pickle.load(f)
    
    with open('testdata/CCDunit_UV1_example.pkl', 'rb') as f:
        CCDunit_UV1=pickle.load(f)        
    CCDitem['CCDunit']=CCDunit_UV1

    table = CCDitem['CCDunit'].get_table(CCDitem)
    ref_table_false = np.load('testdata/IR1_table.npy')
    assert not (table==ref_table_false).all()
    ref_table = np.load('testdata/UV1_table.npy')
    assert (table==ref_table).all()

    image_linear_table,error_flag = inverse_model_table(table,0)
    image_linear_real,error_flag = inverse_model_real(CCDitem,0)
    assert image_linear_table==0.0
    assert np.abs(image_linear_real-image_linear_table)<1e-3
    
    image_linear_table,error_flag = inverse_model_table(table,1e3)
    image_linear_real,error_flag = inverse_model_real(CCDitem,1e3)
    assert np.abs(image_linear_real-image_linear_table)<1e-3

    image_linear_table,error_flag = inverse_model_table(table,10e3)
    image_linear_real,error_flag = inverse_model_real(CCDitem,10e3)
    assert np.abs(image_linear_real-image_linear_table)<1e-3

    image_linear_table,error_flag = inverse_model_table(table,30e3)
    image_linear_real,error_flag = inverse_model_real(CCDitem,30e3)
    assert np.abs(image_linear_real-image_linear_table)<1e-3

def test_error_algebra():
    error_flag_zero= np.zeros([20,30], dtype=np.uint16)
    error_flag_one=np.zeros([20,30], dtype=np.uint16)
    error_flag_one[0,0] = 1

    error_flag=combine_flags([make_binary(error_flag_zero,1), make_binary(error_flag_one,1)])
    assert error_flag[0,0] == '01'
    assert error_flag[0,1] == '00'

    error_flag=combine_flags([make_binary(error_flag_zero,2), make_binary(error_flag_one,1)])
    assert error_flag[0,0] == '001'
    assert error_flag[0,1] == '000'
    
    error_flag_one[0,0] = 3

    error_flag=combine_flags([make_binary(error_flag_zero,2).astype('<U16'), make_binary(error_flag_one,2)])
    assert error_flag[0,0] == '0011'
    assert error_flag[0,1] == '0000'






if __name__ == "__main__":

    test_calibrate()    
    test_readfunctions()
    test_CCDunit()
    test_forward_backward()
    test_non_linearity_fullframe()
    test_non_linearity_binned()
    test_calibrate()
    test_error_algebra()