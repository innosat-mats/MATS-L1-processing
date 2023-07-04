import pytest

from mats_l1_processing.read_and_calibrate_all_files_parallel import main
from mats_l1_processing.instrument import Instrument, CCD
from mats_l1_processing.L1_calibration_functions import inverse_model_real,inverse_model_table,make_binary,combine_flags,desmear,artifact_correction

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
    from mats_l1_processing.read_in_functions import read_all_files_in_root_directory
    from database_generation.experimental_utils import read_all_files_in_protocol, readprotocol
    

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

# def test_forward_backward(): 
#     """
#     This tests the forward and backward calibration. 
#     The backward calibraton should completely reverse everything the forward 
#     calibration has done thus giving back the original image.
#     This test needs a CCDitem and a CCDunit, which are created and saved in 
#     test_reafunctions and test_CCDunit.
    
#     """
#     from mats_l1_processing.forward_model import  forward_and_backward
    
#     with open('testdata/CCDitem_example.pkl', 'rb') as f:
#         CCDitem = pickle.load(f)
    
#     with open('testdata/CCDunit_IR1_example.pkl', 'rb') as f:
#         CCDunit_IR1=pickle.load(f)        
#     CCDitem['CCDunit']=CCDunit_IR1

#     forward_and_backward(CCDitem,  photons=1000, plot=False)

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

    assert combine_flags([0],[1]) == 0
    assert combine_flags([0],[2]) == 0

    assert combine_flags([1],[1]) == 1
    assert combine_flags([0,1],[1,1]) == 2
    assert combine_flags([1,0],[1,1]) == 1

    assert combine_flags([2],[2]) == 2
    assert combine_flags([0,2],[1,2]) == 4

    assert combine_flags([0,3],[1,2]) == 6
    assert combine_flags([1,3],[1,2]) == 7

    A = np.ones((512,2047),dtype=np.int16)
    assert np.all(combine_flags([A,A*3],[1,2])==A*7)
    
def test_channel_quaterion():
    intrument = Instrument("tests/calibration_data_test.toml")
    CCDunit_IR1=intrument.get_CCD("IR1")
    assert np.abs(CCDunit_IR1.get_channel_quaternion()-np.array([-0.705835446710,0.003259749929,0.708320899863,0.008197500630] )).sum()<1e-3
 

def test_desmearing():

    def get_smeared_image(input,rowread_time,nrowskip,exposure_time):
        smeared_array = np.zeros((input_array.shape[0]-nrowskip,input_array.shape[1]))
        smeared_array[0,:] = input_array[0+nrowskip,:]
        
        for j in range(smeared_array.shape[1]):
            for i in range(1,len(smeared_array[:,j])):
                extra_signal = np.sum(input[0:i,j])*rowread_time/exposure_time
                smeared_array[i,j] = input[i+nrowskip,j] + extra_signal

        return smeared_array

    exposure_time = 10
    rowread_time = 0.1
    ncol=10
    nrow=20
    nrskip = 2

    input_array = np.tile(np.linspace(5,3,nrow),(ncol,1)).T*exposure_time
    smeared_array = get_smeared_image(input_array,rowread_time,nrskip,exposure_time)

    corrected_image = desmear(smeared_array,nrskip,rowread_time/exposure_time,input_array[:nrskip,:])
    assert np.sum(corrected_image -  input_array[nrskip:])<1e-9



def test_artifact():

    with open('testdata/artifact_correction/CCDitem_artifact_IR2.pkl', 'rb') as f:
        CCDitem_IR2 = pickle.load(f)

    with open('testdata/artifact_correction/CCDitem_artifact_NADIR.pkl', 'rb') as f:
        CCDitem_nadir = pickle.load(f)

    instrument = Instrument("tests/calibration_data_test.toml")
    CCDunit_IR2=instrument.get_CCD("IR2")
    CCDitem_IR2['CCDunit']=CCDunit_IR2
    CCDunit_nadir=instrument.get_CCD("NADIR")
    CCDitem_nadir['CCDunit']=CCDunit_nadir
    
    
    

    #ccd channel other than NADIR shouldn't be modified
    image = CCDitem_IR2['IMAGE']
    image_no_artifact, error_artifact = artifact_correction(CCDitem_IR2)
    np.testing.assert_allclose(image_no_artifact,image,atol=1e-9)
    expected_error_flag = np.full(np.shape(image),2,dtype=np.uint16)
    np.testing.assert_allclose(expected_error_flag,error_artifact,atol=1e-9)

    
    image = CCDitem_nadir['IMAGE']
    image_expected = np.load('testdata/artifact_correction/image_artifact_corrected.npy')
    error_expected =  np.load('testdata/artifact_correction/artifact_error.npy')
    image_no_artifact, error_artifact = artifact_correction(CCDitem_nadir)
    np.testing.assert_allclose(image_no_artifact,image_expected,atol=1e-9)
    np.testing.assert_allclose(error_expected,error_artifact,atol=1e-9)

    image = np.load('testdata/artifact_correction/image_artifact.npy')
    image_expected = np.load('testdata/artifact_correction/image_artifact_corrected2.npy')
    error_expected =  np.load('testdata/artifact_correction/artifact_error.npy')
    image_no_artifact, error_artifact = artifact_correction(CCDitem_nadir,image)
    np.testing.assert_allclose(image_expected,image_no_artifact,atol=1e-9)
    np.testing.assert_allclose(error_expected,error_artifact,atol=1e-9)


    


def test_calibration_output():
    
    from mats_l1_processing.L1_calibration_functions import (
        get_true_image,
        desmear_true_image,
        subtract_dark,
        flatfield_calibration,
        get_linearized_image,
        artifact_correction
    )
    
   # from mats_l1_processing.read_in_functions import read_CCDitems    
    # directory='testdata/binning_test_20200812/RacFiles_out/'
    # CCDitems = read_CCDitems(directory) 
    # instrument = Instrument("tests/calibration_data_test.toml")
    
    with open('testdata/CCDitem_NSKIP_example.pkl', 'rb') as f:
        CCDitem = pickle.load(f)
    
    
    image_bias_sub,error_flags_bias = get_true_image(CCDitem)

    image_linear,error_flags_linearity = get_linearized_image(CCDitem, image_bias_sub)

    #FIXME: linear image is not tested
    image_desmeared, error_flags_desmear= desmear_true_image(CCDitem, image_bias_sub)

    image_dark_sub, error_flags_dark = subtract_dark(CCDitem, image_desmeared)

    image_calib_nonflipped, error_flags_flatfield = flatfield_calibration(CCDitem, image_dark_sub)


    # no test for image flipping yet
    image_calibrated_flipped = image_calib_nonflipped

    image_calibrated, error_artifact = artifact_correction(CCDitem,image_calibrated_flipped)
    
    with open('testdata/calibration_output.pkl', 'rb') as f:
            [image_bias_sub_old, image_desmeared_old, image_dark_sub_old, image_calib_nonflipped_old, image_no_artifact_old]=pickle.load(f)
    
    assert np.abs(image_bias_sub_old-image_bias_sub).all()<1e-3
    assert np.abs(image_desmeared_old-image_desmeared).all()<1e-3
    assert np.abs(image_dark_sub_old-image_dark_sub).all()<1e-3
    #assert np.abs(image_calib_nonflipped_old-image_calib_nonflipped).all()<1e-3
    assert np.abs(image_no_artifact_old-image_calibrated).all()<1e-3




if __name__ == "__main__":

    test_calibrate()
    test_calibration_output() 
    test_readfunctions()
    test_CCDunit()
    test_non_linearity_fullframe()
    test_non_linearity_binned()
    test_calibrate()
    test_error_algebra()
    test_channel_quaterion()