#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 06:48:10 2020

@author: lindamegner


Functions for data analysis of during the MATS calibration

"""

from mats_l1_processing.read_in_functions import read_CCDitem_from_imgview
import matplotlib.pyplot as plt
from mats_l1_processing.read_in_functions import read_CCDitem


def printvar(my_var):
    my_var_name = [k for k, v in locals().iteritems() if v == my_var][0]
    print(my_var_name, ": ", my_var)


def read_files_in_protocol_as_ItemsUnits(df, imagedir, numberofimagesinunit, read_from):
    # reads files in a protocol as ItemsUnits
    protocolline = list(range(0, len(df), numberofimagesinunit))
    ItemsUnitsList = []
    for ind, line in enumerate(protocolline):
        #   ItemsUnit=ItemsUnitCreate(df[line:line+2],directory+'PayloadImages/')
        if read_from == "imgview":
            ItemsUnit = ItemsUnitCreate(
                df[line : line + numberofimagesinunit], imagedir + "PayloadImages/"
            )
        elif read_from == "rac":
            ItemsUnit = ItemsUnitCreateFromRac(
                df[line : line + numberofimagesinunit], imagedir + "RacFiles_out/"
            )
        else:
            raise Exception("where should it be read from, imgview or rac?")
        ItemsUnitsList.append(ItemsUnit)
    return ItemsUnitsList


def selectimage(df, shutter, imagedir, ExpTime, channel):
    # select a file to create an item unit from
    df_select = df[df.Shutter == shutter]
    ItemsUList = read_files_in_protocol_as_ItemsUnits(df_select, imagedir)
    mylist = list(
        filter(
            lambda x: (
                x.imageItem["TEXPMS"] == ExpTime and x.imageItem["channel"] == channel
            ),
            ItemsUList,
        )
    )
    return mylist[0]


def plot_CCDimage(image, fig, axis, title="", clim=999):
    import numpy as np
    import matplotlib.pyplot as plt

    sp = axis.imshow(image, cmap=plt.cm.gray)
    if clim == 999:
        mean = np.mean(image)
        std = np.std(image)
        sp.set_clim([mean - 2 * std, mean + 2 * std])
    else:
        sp.set_clim(clim)
    fig.colorbar(sp, ax=axis)
    axis.set_title(title)
    return sp


def plot_simple(filename, clim=999):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    pic = np.float64(Image.open(filename))  # read image
    plt.imshow(pic)
    if clim == 999:
        mean = pic.mean()
        std = pic.std()
        plt.clim([mean - 2 * std, mean + 2 * std])
    else:
        plt.clim(clim)
    plt.colorbar()


def readandsubtractdark(dirname, imageID, dark1ID, dark2ID="999", multiplydark=1):
    # Takes directory and PicID of image and dark picture(s) and subtracts them

    image = read_CCDitem_from_imgview(dirname, imageID)
    dark1 = read_CCDitem_from_imgview(dirname, dark1ID)
    if dark2ID == "999":
        imagenew = image["IMAGE"] - multiplydark * dark1["IMAGE"]
    else:
        dark2 = read_CCDitem_from_imgview(dirname, dark2ID)
        imagenew = (
            image["IMAGE"] - multiplydark * (dark1["IMAGE"] + dark2["IMAGE"]) / 2.0
        )
    return imagenew


def plotCCDitem(CCDitem, fig, axis, title="", clim=999):

    pic = CCDitem["IMAGE"]
    sp = axis.pcolormesh(pic, cmap=plt.cm.jet)
    axis.set_title(title)
    if clim == 999:
        mean = pic.mean()
        std = pic.std()
        sp.set_clim([mean - 2 * std, mean + 2 * std])
    else:
        sp.set_clim(clim)

    fig.colorbar(sp, ax=axis)

    return sp


def diffplot(image1, image2, title1, title2, clim=999, climdiff=999):
    from LindasCalibrationFunctions import plot_CCDimage

    fig, ax = plt.subplots(3, 1)
    diffimg = image1 - image2
    if clim == 999:
        plot_CCDimage(image1, fig, ax[0], title=title1)
        plot_CCDimage(image2, fig, ax[1], title=title2)
    else:
        plot_CCDimage(image1, fig, ax[0], title=title1, clim=clim)
        plot_CCDimage(image2, fig, ax[1], title=title2, clim=clim)
    if climdiff == 999:
        plot_CCDimage(diffimg, fig, ax[2], title="pic 1-pic2")
    else:
        plot_CCDimage(diffimg, fig, ax[2], title="pic 1-pic2", clim=climdiff)
    return fig


def UniqueValuesInKey(listofdicts, keystr):
    uniqueValues = list()
    for x in listofdicts:
        if x[keystr] not in uniqueValues:
            uniqueValues.append(x[keystr])
    return uniqueValues


def matrixmean(mat1, mat2, mat3="none", mat4="none"):
    if mat3 == "none" and mat4 == "none":
        matav = (mat1 + mat2) / 2.0
    elif mat3 == "none":
        matav = (mat1 + mat2 + mat4) / 3.0
    elif mat4 == "none":
        matav = (mat1 + mat2 + mat3) / 3.0
    else:
        matav = (mat1 + mat2 + mat3 + mat4) / 4.0
    return matav


class ItemsUnitCreate:
    def __init__(self, df, dirname):
        import numpy as np

        self.df = df
        df_B = df[df.DarkBright == "B"]
        df_D = df[df.DarkBright == "D"]

        self.imageItem = read_CCDitem_from_imgview(dirname, df_B.PicID.iloc[0])

        # =============================================================================
        #         # imageitems=[]
        #         # for i, dfitem in enumerate(df_B):
        #         #     imageitems.append(read_CCDitem_from_imgview(dirname,dfitem))
        #         # darkitems=[]
        #         # for i, dfitem in enumerate(df_D):
        #         #     darkitems.append(read_CCDitem_from_imgview(dirname,dfitem))
        #         # self.image=meanimage(imageitems)
        #         # self.dark=meanimage(darkitems)
        # =============================================================================
        if len(df_B) == 1:
            self.image = self.imageItem["IMAGE"]
        elif len(df_B) == 2:
            self.image1Item = read_CCDitem_from_imgview(dirname, df_B.PicID.iloc[0])
            self.image2Item = read_CCDitem_from_imgview(dirname, df_B.PicID.iloc[1])
            self.image = matrixmean(self.image1Item["IMAGE"], self.image2Item["IMAGE"])
        else:
            raise Exception(str(len(df_B)) + " brightimage(s) in dataframe")

        if len(df_D) == 1:
            self.darkItem = read_CCDitem_from_imgview(dirname, df_D.PicID.iloc[0])
            self.dark = self.darkItem["IMAGE"]
        elif len(df_D) == 2:
            self.dark1Item = read_CCDitem_from_imgview(dirname, df_D.PicID.iloc[0])
            self.dark2Item = read_CCDitem_from_imgview(dirname, df_D.PicID.iloc[1])
            self.dark = matrixmean(self.dark1Item["IMAGE"], self.dark2Item["IMAGE"])
        elif len(df_D) == 3:
            self.dark1Item = read_CCDitem_from_imgview(dirname, df_D.PicID.iloc[0])
            self.dark2Item = read_CCDitem_from_imgview(dirname, df_D.PicID.iloc[1])
            self.dark3Item = read_CCDitem_from_imgview(dirname, df_D.PicID.iloc[2])
            self.dark = matrixmean(
                self.dark1Item["IMAGE"],
                self.dark2Item["IMAGE"],
                self.dark3Item["IMAGE"],
            )
        else:
            raise Exception(str(len(df_D)) + " dark pictures in dataframe")
        self.subpic = self.image - self.dark

    def plot(self, fig, axis, whichpic=2, title="", clim=999):
        # whichpic 0 is image, whichpic 1 is dark  whichpic 2 is subpic

        if whichpic == 0:
            pic = self.dark
        elif whichpic == 1:
            pic = self.image
        elif whichpic == 2:
            pic = self.subpic
        else:
            raise Exception("whichpic must be 1 2 or 3")

        sp = axis.pcolormesh(pic, cmap=plt.cm.jet)
        axis.set_title(title)
        if clim == 999:
            #            sp.set_clim([0,4000])
            mean = pic.mean()
            std = pic.std()
            sp.set_clim([mean - 1 * std, mean + 1 * std])
        else:
            sp.set_clim(clim)

        fig.colorbar(sp, ax=axis)

        return sp


class ItemsUnitCreateFromRac:
    def __init__(self, df, dirname):
        import numpy as np

        self.df = df
        df_B = df[df.DarkBright == "B"]
        df_D = df[df.DarkBright == "D"]

        self.imageItem = read_CCDitem(dirname, df_B.PicID.iloc[0])

        # =============================================================================
        #         # imageitems=[]
        #         # for i, dfitem in enumerate(df_B):
        #         #     imageitems.append(read_CCDitem_from_imgview(dirname,dfitem))
        #         # darkitems=[]
        #         # for i, dfitem in enumerate(df_D):
        #         #     darkitems.append(read_CCDitem_from_imgview(dirname,dfitem))
        #         # self.image=meanimage(imageitems)
        #         # self.dark=meanimage(darkitems)
        # =============================================================================
        if len(df_B) == 1:
            self.image = self.imageItem["IMAGE"]
        elif len(df_B) == 2:
            self.image1Item = read_CCDitem(dirname, df_B.PicID.iloc[0])
            self.image2Item = read_CCDitem(dirname, df_B.PicID.iloc[1])
            self.image = matrixmean(self.image1Item["IMAGE"], self.image2Item["IMAGE"])
        else:
            raise Exception(str(len(df_B)) + " brightimage(s) in dataframe")

        if len(df_D) == 1:
            self.darkItem = read_CCDitem(dirname, df_D.PicID.iloc[0])
            self.dark = self.darkItem["IMAGE"]
        elif len(df_D) == 2:
            self.dark1Item = read_CCDitem(dirname, df_D.PicID.iloc[0])
            self.dark2Item = read_CCDitem(dirname, df_D.PicID.iloc[1])
            self.dark = matrixmean(self.dark1Item["IMAGE"], self.dark2Item["IMAGE"])
        elif len(df_D) == 3:
            self.dark1Item = read_CCDitem(dirname, df_D.PicID.iloc[0])
            self.dark2Item = read_CCDitem(dirname, df_D.PicID.iloc[1])
            self.dark3Item = read_CCDitem(dirname, df_D.PicID.iloc[2])
            self.dark = matrixmean(
                self.dark1Item["IMAGE"],
                self.dark2Item["IMAGE"],
                self.dark3Item["IMAGE"],
            )
        else:
            raise Exception(str(len(df_D)) + " dark pictures in dataframe")
        self.subpic = self.image - self.dark

    def plot(self, fig, axis, whichpic=2, title="", clim=999):
        # whichpic 0 is image, whichpic 1 is dark  whichpic 2 is subpic

        if whichpic == 0:
            pic = self.dark
        elif whichpic == 1:
            pic = self.image
        elif whichpic == 2:
            pic = self.subpic
        else:
            raise Exception("whichpic must be 1 2 or 3")

        sp = axis.imshow(pic, cmap=plt.cm.jet)
        axis.set_title(title)
        if clim == 999:
            #            sp.set_clim([0,4000])
            mean = pic.mean()
            std = pic.std()
            sp.set_clim([mean - 1 * std, mean + 1 * std])
        else:
            sp.set_clim(clim)

        fig.colorbar(sp, ax=axis)

        return sp
