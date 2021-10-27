import pytest

from mats_l1_processing.read_and_calibrate_all_files import main

__author__ = "Ole Martin Christensen"
__copyright__ = "Ole Martin Christensen"
__license__ = "MIT"


def test_calibrate():
    main("testdata/RacFiles_out/", "tests/calibration_data_test.toml")


def test_plot():
    main(
        "testdata/RacFiles_out/",
        "tests/calibration_data_test.toml",
        calibrate=False,
        plot=True,
    )


def test_calibrate_and_plot():
    main(
        "testdata/RacFiles_out/",
        "tests/calibration_data_test.toml",
        calibrate=True,
        plot=True,
    )

def test_readfunctions():
    from mats_l1_processing.read_in_functions import readprotocol, read_all_files_in_root_directory
    from mats_l1_processing.LindasCalibrationFunctions import read_all_files_in_protocol


    directory='testdata/210215OHBLimbImage/'
    protocol='protocol_dark_bright_100um_incl_IR3.txt'


    read_from="rac" 
    df_protocol=readprotocol(directory+protocol)

    df_bright=df_protocol[df_protocol.DarkBright=='B']
    CCDitems=read_all_files_in_protocol(df_bright, read_from,directory)

    CCDitems=read_all_files_in_root_directory(read_from,directory)
    
    read_from="imgview" 
    CCDitems=read_all_files_in_root_directory(read_from,directory)
    