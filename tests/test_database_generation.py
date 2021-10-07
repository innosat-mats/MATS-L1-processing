import pytest

from database_generation.flatfield import make_flatfield
from database_generation.linearity import make_linearity

__author__ = "Ole Martin Christensen"
__copyright__ = "Ole Martin Christensen"
__license__ = "MIT"


calibration_file = "tests/calibration_data_test.toml"


def test_generate_flatfield():

    flatfield_morphed = make_flatfield("IR1", "HSM", calibration_file, plot=False)


def test_get_linearity():
    make_linearity([1], calibration_file, plot=False)
