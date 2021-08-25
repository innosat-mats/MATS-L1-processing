import pytest

from mats_l1_processing.read_and_calibrate_all_files import main

__author__ = "Ole Martin Christensen"
__copyright__ = "Ole Martin Christensen"
__license__ = "MIT"


def test_calibrate():
    main("/home/olemar/Projects/MATS/MATS-L1-processing/testdata/RacFiles_out/")


def test_plot():
    main(
        "/home/olemar/Projects/MATS/MATS-L1-processing/testdata/RacFiles_out/",
        calibrate=False,
        plot=True,
    )


def test_calibrate_and_plot():
    main(
        "/home/olemar/Projects/MATS/MATS-L1-processing/testdata/RacFiles_out/",
        calibrate=True,
        plot=True,
    )
