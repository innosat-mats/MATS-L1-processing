from mats_l1_processing.grid_image import get_shift


def test_shift():
    pass
    # CCDitem = dict()
    # CCDitem["channel"] = "IR1"
    # CCDitem["flipped"] = False
    # CCDitem["NCSKIP"] = 0
    # CCDitem["NROWSKIP"] = 0
    # CCDitem["NCBIN FPGAColumns"] = 1
    # CCDitem["NCBIN CCDColumns"] = 1
    # CCDitem["NRBIN"] = 1
    # assert get_shift(CCDitem) == (75, 47)

    # CCDitem["NCBIN FPGAColumns"] = 2
    # CCDitem["NCBIN CCDColumns"] = 2
    # CCDitem["NRBIN"] = 2
    # assert get_shift(CCDitem) == (75, 47)
    
    # CCDitem["NCSKIP"] = 1
    # assert get_shift(CCDitem,skip_comp=True) == (76, 47)

    # CCDitem["NROWSKIP"] = 1
    # assert get_shift(CCDitem,skip_comp=True) == (76, 48)


if __name__ == "__main__":

    test_shift()
