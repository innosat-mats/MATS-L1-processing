import pytest

from mats_l1_processing.grid_image import grid_image,get_shift
from mats_l1_processing.pointing import pix_deg
from numpy import abs

def test_shift():
    CCDitem = dict()
    CCDitem["channel"] = "IR1"
    CCDitem["flipped"] = False
    CCDitem["NCSKIP"] = 0
    CCDitem["NROWSKIP"] = 0
    CCDitem["NCBIN FPGAColumns"] = 1
    CCDitem["NCBIN CCDColumns"] = 1
    CCDitem["NRBIN"] = 1
    assert get_shift(CCDitem) == (75,47)

    CCDitem["NCBIN FPGAColumns"] = 2
    CCDitem["NCBIN CCDColumns"] = 2
    CCDitem["NRBIN"] = 2
    assert get_shift(CCDitem) == (75,47)
    
    CCDitem["NCSKIP"] = 1
    assert get_shift(CCDitem,skip_comp=True) == (76,47)

    CCDitem["NROWSKIP"] = 1
    assert get_shift(CCDitem,skip_comp=True) == (76,48)
    
    return

def test_pixdeg():


    ccditem = {}
    ccditem['NCSKIP'] = 0
    ccditem['NCBIN CCDColumns'] = 40
    ccditem['NRSKIP'] = 0
    ccditem['NRBIN'] = 2
    ccditem['NCOL'] = 20
    ccditem['CCDSEL'] = 2

    a,b = pix_deg(ccditem, 7, 9)
    assert(abs(a - -2.141871030776746)<1e-6)
    assert(abs(b - -0.7033725490196079)<1e-6)

    ccditem = {}
    ccditem['NCSKIP'] = 0
    ccditem['NCBINCCDColumns'] = 40
    ccditem['NRSKIP'] = 0
    ccditem['NRBIN'] = 2
    ccditem['NCOL'] = 20
    ccditem['CCDSEL'] = 2

    a,b = (pix_deg(ccditem, 70, 90))
    assert(abs(a - 5.318412310698583)<1e-6)
    assert(abs(b - -0.22054901960784315)<1e-6)


    ccditem = {}
    ccditem['NCSKIP'] = 200
    ccditem['NCBINCCDColumns'] = 1
    ccditem['NRSKIP'] = 50
    ccditem['NRBIN'] = 6
    ccditem['NCOL'] = 30
    ccditem['CCDSEL'] = 4

    a,b = pix_deg(ccditem, 50, 2)
    assert(abs(a - -2.288412310698583)<1e-6)
    assert(abs(b - -0.5662745098039216)<1e-6)
    

    ccditem = {}
    ccditem['NCSKIP'] = 200
    ccditem['NCBINCCDColumns'] = 1
    ccditem['NRSKIP'] = 50
    ccditem['NRBIN'] = 6
    ccditem['NCOL'] = 30
    ccditem['CCDSEL'] = 3

    a,b = pix_deg(ccditem, 50, 2)
    assert(abs(a - 2.498602833414753)<1e-6)
    assert(abs(b - -0.5662745098039216)<1e-6)
    

if __name__ == "__main__":

    test_shift()
    test_pixdeg()