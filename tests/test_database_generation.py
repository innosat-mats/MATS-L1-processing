import pytest

from database_generation.flatfield import make_flatfield
from database_generation.linearity import make_linearity
from database_generation.linearity import get_threshold
import numpy as np

__author__ = "Ole Martin Christensen"
__copyright__ = "Ole Martin Christensen"
__license__ = "MIT"


calibration_file = "tests/calibration_data_test.toml"


def test_generate_flatfield():

    flatfield_morphed = make_flatfield("IR1", "HSM", calibration_file, plot=False)


def test_get_linearity():
    make_linearity([1], calibration_file, plot=False)

def test_get_threshold():
    p=np.array([0.98,-0.0000047, 6916])
    assert np.abs((get_threshold(p,0.8, fittype='threshold2')-30597.91319148936)) < 0.1 