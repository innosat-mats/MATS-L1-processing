import pytest

from mats_l1_processing.read_and_calibrate_all_files_parallel import main
from mats_l1_processing.instrument import Instrument, CCD, Photometer
from mats_l1_processing import photometer
import pandas as pd
from mats_l1_processing.L1_calibration_functions import inverse_model_real,inverse_model_table,make_binary,combine_flags,desmear,artifact_correction, correct_single_events,correct_hotpixels, padlastrowsofimage
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
    
    with open('testdata/CCD_items_in_orbit_UVIR.pkl', 'rb') as f:
        CCDitems = pickle.load(f)

    start_date  = np.datetime64(CCDitems[0]['EXP Date'],'s').astype(datetime)
    end_date  = np.datetime64(CCDitems[-1]['EXP Date'],'s').astype(datetime)

    instrument = Instrument("tests/calibration_data_test.toml",start_datetime=start_date,end_datetime=end_date)    


    #IR1
    i = 5
    images = L1_calibrate(CCDitems[i], instrument,return_steps=True)
    
def make_calibration_plots(images,datechannel):

    (image_lsb,
     image_se_corrected, 
     image_hot_pixel_corrected, 
     image_bias_sub, image_linear,
     image_desmeared, 
     image_dark_sub, 
     image_flatfielded, 
     image_flipped, 
     image_calibrated, 
     errors,
    ) = images

    plot_calib_step(image_lsb,image_se_corrected,'SE_correction_' + datechannel,errors)
    plot_calib_step(image_se_corrected,image_hot_pixel_corrected,'hot-pixel_correction' + datechannel,errors)
    plot_calib_step(image_hot_pixel_corrected,image_bias_sub,'bias_subtraction' + datechannel,errors)
    plot_calib_step(image_bias_sub,image_linear,'linearization' + datechannel,errors,divide=True)
    plot_calib_step(image_linear,image_desmeared,'desmear' + datechannel,errors)
    plot_calib_step(image_desmeared,image_dark_sub,'dark_subtraction' + datechannel,errors)
    plot_calib_step(image_dark_sub,image_flatfielded,'flatfielding' + datechannel,errors,divide=True)
    plot_calib_step(image_flatfielded,image_flipped,'flipping' + datechannel,errors)
    #plot_calib_step(image_flipped,image_calibrated,'image_calibrated',errors)

    return

def plot_calib_step(step_1,step2,title,error,divide=False):

    error = np.zeros(np.shape(step_1))

    fig, axs = plt.subplots(1,3)
    fig.suptitle(title, fontsize=16)
    sc = axs[0].imshow(step_1, vmin=np.mean(step_1)-np.std(step_1), vmax=np.mean(step_1)+np.std(step_1),origin='lower')
    cbar = fig.colorbar(sc, ax=axs[0], orientation='vertical')
    sc = axs[1].imshow(step2, vmin=np.mean(step2)-np.std(step2), vmax=np.mean(step2)+np.std(step2),origin='lower')
    cbar = fig.colorbar(sc, ax=axs[1], orientation='vertical')
    if divide:
        sc = axs[2].imshow(step2/step_1,vmin=np.median(step2/step_1)-np.std(step2/step_1), vmax=np.median(step2/step_1)+np.std(step2/step_1),origin='lower')
    else:
        sc = axs[2].imshow(step2-step_1,origin='lower')
    cbar = fig.colorbar(sc, ax=axs[2], orientation='vertical')
    # axs[3].imshow(error,origin='lower')
    # cbar = fig.colorbar(sc, ax=axs[3], orientation='vertical')

    plt.tight_layout()
    plt.savefig(title + '.png')
    plt.show()


    return

def test_calibrate_plots():
    from mats_l1_processing.read_in_functions import read_CCDitems 
    instrument = Instrument("tests/calibration_data_test.toml")    

    with open('testdata/CCD_items_in_orbit_UVIR.pkl', 'rb') as f:
        CCDitems = pickle.load(f)


    #IR1
    i = 5
    images = L1_calibrate(CCDitems[i], instrument,return_steps=True)
    datechannel = str(CCDitems[i]["TMHeaderTime"])[0:10] + '_' + CCDitems[i]["channel"]
    make_calibration_plots(images,datechannel)

    #IR2
    i = 2
    images = L1_calibrate(CCDitems[i], instrument,return_steps=True)
    datechannel = str(CCDitems[i]["TMHeaderTime"])[0:10] + '_' + CCDitems[i]["channel"]
    make_calibration_plots(images,datechannel)

    #IR3
    i = 1
    images = L1_calibrate(CCDitems[i], instrument,return_steps=True)
    datechannel = str(CCDitems[i]["TMHeaderTime"])[0:10] + '_' + CCDitems[i]["channel"]
    make_calibration_plots(images,datechannel)

    #IR4
    i = 6
    images = L1_calibrate(CCDitems[i], instrument,return_steps=True)
    datechannel = str(CCDitems[i]["TMHeaderTime"])[0:10] + '_' + CCDitems[i]["channel"]
    make_calibration_plots(images,datechannel)
    
    #UV1
    i = 3
    images = L1_calibrate(CCDitems[i], instrument,return_steps=True)
    datechannel = str(CCDitems[i]["TMHeaderTime"])[0:10] + '_' + CCDitems[i]["channel"]
    make_calibration_plots(images,datechannel)

    #UV2
    i = 4
    images = L1_calibrate(CCDitems[i], instrument,return_steps=True)
    datechannel = str(CCDitems[i]["TMHeaderTime"])[0:10] + '_' + CCDitems[i]["channel"]
    make_calibration_plots(images,datechannel)


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
    
    
    
    with open('testdata/CCDitem_NSKIP_example.pkl', 'rb') as f:
        CCDitem = pickle.load(f)
    
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
   
   
    with open('testdata/calibration_output.pkl', 'rb') as f:
            [image_lsb_old,image_bias_sub_old,image_desmeared_old,image_dark_sub_old,image_calib_nonflipped_old,image_calib_flipped_old,image_calibrated_old]=pickle.load(f) 
    
    # assert (np.abs(image_bias_sub_old-image_bias_sub)<1e-3).all()
    # assert (np.abs(image_desmeared_old-image_desmeared)<1e-3).all()
    # assert (np.abs(image_dark_sub_old-image_dark_sub)<1e-3).all()
    # assert (np.abs(image_calib_nonflipped_old-image_calib_nonflipped)<1e-3).all()
    # assert (np.abs(image_calib_flipped_old-image_calib_flipped)<1e-3).all() 
    # assert (np.abs(image_calibrated_old-image_calibrated)<1e-3).all()


def photometer_assertion(photometer_data,photometer_data_out,i,i_out,j):
    try: 
        assert(np.any(photometer_data.iloc[j][i]== photometer_data_out.iloc[j][i_out]))
    except AssertionError:
        if np.isnan(photometer_data_out.iloc[j][i_out]):
            pass           
        elif np.abs(((photometer_data.iloc[j][i] - photometer_data_out.iloc[j][i_out])/photometer_data_out.iloc[j][i_out]))<0.015:
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
            
def test_se_correction():
    
    with open('testdata/CCD_items_in_orbit_nightglow_example.pkl', 'rb') as f:
        CCDitems = pickle.load(f)
    
    CCDitem = CCDitems[4]
    instrument = Instrument("tests/calibration_data_test.toml")
    CCDitem["CCDunit"] =instrument.get_CCD(CCDitem["channel"])
    se_corrected,se_mask = correct_single_events(CCDitem,CCDitem['IMAGE'])

    # plt.figure()
    # plt.imshow(CCDitem['IMAGE'],origin='lower')
    # plt.axis('auto')
    # plt.colorbar()
    # plt.title('original image')
    # plt.show()

    # plt.figure()
    # plt.imshow(se_corrected,origin='lower')
    # plt.axis('auto')
    # plt.colorbar()
    # plt.title('corrected image')
    # plt.show()

    # plt.figure()
    # plt.imshow(CCDitem['IMAGE']-se_corrected,origin='lower')
    # plt.axis('auto')
    # plt.colorbar()
    # plt.title('difference')
    # plt.show()

    return

def test_hp_correction():
    
    with open('testdata/CCD_items_in_orbit_nightglow_example.pkl', 'rb') as f:
        CCDitems = pickle.load(f)

    CCDitem = CCDitems[7]
    instrument = Instrument("tests/calibration_data_test.toml")
    CCDitem["CCDunit"] =instrument.get_CCD(CCDitem["channel"])
    hot_pixel_corrected,hot_pixel_mask = correct_hotpixels(CCDitem,CCDitem['IMAGE'])
    test_data = np.load('hot_pixel_corrected_7.npz')
    assert np.equal(hot_pixel_corrected,test_data["hot_pixel_corrected"]).all() 
    assert np.equal(hot_pixel_mask,test_data["hot_pixel_mask"]).all()


    CCDitem = CCDitems[4]
    instrument = Instrument("tests/calibration_data_test.toml")
    CCDitem["CCDunit"] =instrument.get_CCD(CCDitem["channel"])
    hot_pixel_corrected,hot_pixel_mask = correct_hotpixels(CCDitem,CCDitem['IMAGE'])
    #test_data = np.load('hot_pixel_corrected_4.npz')
    #assert np.equal(hot_pixel_corrected,test_data["hot_pixel_corrected"]).all() 
    #assert np.equal(hot_pixel_mask,test_data["hot_pixel_mask"]).all()

    # plt.figure()
    # plt.imshow(CCDitem['IMAGE'],origin='lower')
    # plt.axis('auto')
    # plt.colorbar()
    # plt.title('original image')
    # plt.show()

    # plt.figure()
    # plt.imshow(hot_pixel_corrected,origin='lower')
    # plt.axis('auto')
    # plt.colorbar()
    # plt.title('corrected image')
    # plt.show()

    # plt.figure()
    # plt.imshow(CCDitem['IMAGE']-hot_pixel_corrected,origin='lower')
    # plt.axis('auto')
    # plt.colorbar()
    # plt.title('difference')
    # plt.show()

    return 

def test_image_padding():
    with open('testdata/flatfield_IR2_HSM.pkl', 'rb') as f:
        image = pickle.load(f)
    [rows, colums]=image.shape
    #print('shape of image',image.shape)
    nrow=3
    image_padded=padlastrowsofimage(image,nrow)
    assert np.shape(image_padded)==(rows+nrow,colums), "Image padding shape mismatch"
    assert np.all(image_padded[512:,:]==image_padded[511,:]), "Image padding values mismatch"
    #print('shape of padded image',image_padded.shape)
if __name__ == "__main__":
    
    test_calibrate()
    # test_calibrate_plots()
    # test_error_algebra()
    # test_channel_quaterion()
    # test_photometer()
    # test_hp_correction()
    # test_se_correction()
    # test_image_padding()
# %%
