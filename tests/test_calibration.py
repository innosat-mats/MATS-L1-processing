import pytest

from mats_l1_processing.read_and_calibrate_all_files_parallel import main
from mats_l1_processing.instrument import Instrument, CCD, Photometer
from mats_l1_processing import photometer
import pandas as pd
from mats_l1_processing.L1_calibration_functions import inverse_model_real,make_binary,combine_flags,desmear,artifact_correction, correct_single_events,correct_hotpixels, padlastrowsofimage
from mats_l1_processing.L1_calibrate import L1_calibrate
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

__author__ = "Ole Martin Christensen"
__copyright__ = "Ole Martin Christensen"
__license__ = "MIT"


def test_calibrate():

    #Test to check that the code is running
    
    with open('testdata/CCD_items_in_orbit_nightglow_example.pkl', 'rb') as f:
        CCDitems = pickle.load(f)

    start_date  = pd.Timestamp(CCDitems[0]['EXP Date']).floor('s').to_pydatetime().replace(tzinfo=None)
    end_date  = pd.Timestamp(CCDitems[-1]['EXP Date']).floor('s').to_pydatetime().replace(tzinfo=None)

    instrument = Instrument("tests/calibration_data_test.toml",start_datetime=start_date,end_datetime=end_date)    


    #IR1
    i = 5
    images = L1_calibrate(CCDitems[i], instrument,return_steps=True)


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
    assert np.abs(CCDunit_IR1.get_channel_quaternion()-np.array([-0.7057631884537752,0.0013168893327113714,0.7084190827489855,0.006244263217732613] )).sum()<1e-3
 

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



def test_artifact(): #Test fails @Louis test data not uploaded to box

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
        artifact_correction,
        flip_image
    )
    
    
    
    with open('testdata/CCD_items_in_orbit_nightglow_example.pkl', 'rb') as f:
        CCDitems = pickle.load(f)
    
    CCDitem = CCDitems[1]
    instrument = Instrument("tests/calibration_data_test.toml")
    CCDitem["CCDunit"] =instrument.get_CCD(CCDitem["channel"])

    # removing the NADIR artifact (subject to change)
    image_lsb = CCDitem['IMAGE']
    

    # IF CHANNEL NOT NADIR check is the artifact correction is not applied

    image_bias_sub,error_flags_bias = get_true_image(CCDitem,image_lsb)

    image_linear,error_flags_linearity = get_linearized_image(CCDitem, image_bias_sub)

    #FIXME: linear image is not tested
    image_desmeared, error_flags_desmear= desmear_true_image(CCDitem, image_bias_sub)

    image_dark_sub, error_flags_dark = subtract_dark(CCDitem, image_desmeared)

    image_calib_nonflipped, error_flags_flatfield = flatfield_calibration(CCDitem, image_dark_sub)


    # no test for image flipping yet
    image_calib_flipped = flip_image(CCDitem, image_calib_nonflipped)

    image_calibrated, error_artifact = artifact_correction(CCDitem,image_calib_flipped)
   
   
    # with open('testdata/calibration_output.pkl', 'rb') as f:
    #         [image_lsb_old,image_bias_sub_old,image_desmeared_old,image_dark_sub_old,image_calib_nonflipped_old,image_calib_flipped_old,image_calibrated_old]=pickle.load(f) 
    
    # assert (np.abs(image_bias_sub_old-image_bias_sub)<1e-3).all()
    # assert (np.abs(image_desmeared_old-image_desmeared)<1e-3).all()
    # assert (np.abs(image_dark_sub_old-image_dark_sub)<1e-3).all()
    # assert (np.abs(image_calib_nonflipped_old-image_calib_nonflipped)<1e-3).all()
    # assert (np.abs(image_calib_flipped_old-image_calib_flipped)<1e-3).all() 
    # assert (np.abs(image_calibrated_old-image_calibrated)<1e-3).all()


def photometer_assertion(photometer_data,photometer_data_out,i,i_out,j):
    try: 
        assert(np.any(photometer_data.iloc[j].iloc[i]== photometer_data_out.iloc[j].iloc[i_out]))
    except AssertionError:
        if np.isnan(photometer_data_out.iloc[j].iloc[i_out]):
            pass
        elif np.abs(((photometer_data.iloc[j].iloc[i] - photometer_data_out.iloc[j].iloc[i_out])/photometer_data_out.iloc[j].iloc[i_out]))<0.015:
            pass
        elif (photometer_data.iloc[j].index[i] == 'pmTEXPMS'):
            pass
        elif (photometer_data.iloc[j].index[i] == 'pmAband_Sig') and (photometer_data.iloc[j].pmAband_Sig_bit < 2):
            pass
        elif (photometer_data.iloc[j].index[i] == 'pmBkg_Sig') and (photometer_data.iloc[j].pmBkg_Sig_bit < 2):
            pass
        else:
            raise AssertionError

    return
            

def test_photometer():
    
    photometer_calib = Photometer("tests/calibration_data_test.toml")
    with open('testdata/photometer_test_data_in.pkl', 'rb') as f:
        photometer_data = pickle.load(f)
    
    photometer.calibrate_pm(photometer_data,photometer_calib)

    with open('testdata/photometer_test_data_out.pkl', 'rb') as f:
        photometer_data_out = pickle.load(f)
    
    for j in range(0,len(photometer_data_out),100):
        for i in range(len(photometer_data_out.iloc[j])): 
            if i < 29:
                photometer_assertion(photometer_data,photometer_data_out,i,i,j)
            elif i < 32:
                photometer_assertion(photometer_data,photometer_data_out,i+1,i,j)
            else:
                photometer_assertion(photometer_data,photometer_data_out,i+2,i,j)


test_photometer()