import pytest

from mats_l1_processing.read_and_calibrate_all_files import main

__author__ = "Ole Martin Christensen"
__copyright__ = "Ole Martin Christensen"
__license__ = "MIT"


def test_main():
    main("/home/olemar/Projects/MATS/MATS-L1-processing/data/")
