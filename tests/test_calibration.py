import pytest

from mats_l1_processing.read_and_calibrate_all_files import main

__author__ = "Ole Martin Christensen"
__copyright__ = "Ole Martin Christensen"
__license__ = "MIT"


def test_calibrate():
    main("testdata/RacFiles_out/", "tests/calibration_data_test.toml")


def test_plot():
    main(
        "testdata/RacFiles_out/", "tests/calibration_data_test.toml",
        calibrate=False,
        plot=True,
    )


def test_calibrate_and_plot():
    main(
        "testdata/RacFiles_out/", "tests/calibration_data_test.toml",
        calibrate=True,
        plot=True,
    )
