from pytest import approx  # type: ignore

from mats_l1_processing.pointing import pix_deg


def test_pixdeg():

    ccditem = {}
    ccditem['NCSKIP'] = 0
    ccditem['NCBIN CCDColumns'] = 40
    ccditem['NRSKIP'] = 0
    ccditem['NRBIN'] = 2
    ccditem['NCOL'] = 20
    ccditem['CCDSEL'] = 2

    a, b = pix_deg(ccditem, 7, 9)
    assert(a == approx(-2.1394290161657805, abs=1e-6))
    assert(b == approx(-0.6995209721539268, abs=1e-6))    

    ccditem = {}
    ccditem['NCSKIP'] = 0
    ccditem['NCBINCCDColumns'] = 40
    ccditem['NRSKIP'] = 0
    ccditem['NRBIN'] = 2
    ccditem['NCOL'] = 20
    ccditem['CCDSEL'] = 2

    a, b = (pix_deg(ccditem, 70, 90))
    assert(a == approx(5.29965319095335, abs=1e-6))
    assert(b == approx(-0.2193511489861099, abs=1e-6)) 


    ccditem = {}
    ccditem['NCSKIP'] = 200
    ccditem['NCBINCCDColumns'] = 1
    ccditem['NRSKIP'] = 50
    ccditem['NRBIN'] = 6
    ccditem['NCOL'] = 30
    ccditem['CCDSEL'] = 4

    a,b = pix_deg(ccditem, 50, 2)
    assert(a == approx(-2.285652940019572, abs=1e-6))
    assert(b == approx(-0.5631835091518915, abs=1e-6)) 
    

    ccditem = {}
    ccditem['NCSKIP'] = 200
    ccditem['NCBINCCDColumns'] = 1
    ccditem['NRSKIP'] = 50
    ccditem['NRBIN'] = 6
    ccditem['NCOL'] = 30
    ccditem['CCDSEL'] = 3

    a, b = pix_deg(ccditem, 50, 2)
    assert(a == approx(2.4953357893889208, abs=1e-6))
    assert(b == approx(-0.5631835091518915, abs=1e-6)) 
    

if __name__ == "__main__":

    test_pixdeg()
