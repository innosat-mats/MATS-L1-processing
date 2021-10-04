import pytest

from database_generation.flatfield import make_flatfield

__author__ = "Ole Martin Christensen"
__copyright__ = "Ole Martin Christensen"
__license__ = "MIT"


def test_generate_flatfield():

    flatfield_morphed=make_flatfield('IR1', 'HSM', "tests/calibration_data_test.toml",plot=False)
