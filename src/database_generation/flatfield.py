#%%

import numpy as np
from PIL import Image
from mats_l1_processing.instrument import CCD
from database_generation.experimental_utils import plot_CCDimage
import matplotlib.pyplot as plt

from pathlib import Path
import toml



def read_flatfield(CCDunit, mode, flatfield_directory):
    from mats_l1_processing.items_units_functions import (
        read_files_in_protocol_as_ItemsUnits,
    )
    from database_generation.experimental_utils import readprotocol

    if mode == 'HSM': 
        directory = flatfield_directory
        # protocol='flatfields_200330_SigMod1_LMprotocol.txt'
        protocol = "readin_flatfields_SigMod1.txt"

    elif mode == 'LSM':  # LSM
        directory = flatfield_directory

        protocol = "readin_flatfields_SigMod0.txt"
    else:
        print("Undefined mode")

    read_from = "rac"
    df_protocol = readprotocol(directory + protocol)
    # df_only2 = df_protocol[(df_protocol.index-2) % 3 != 0]

    # The below reads all images in protocol - very inefficient. Should be only one file read in LM200810
    CCDItemsUnits = read_files_in_protocol_as_ItemsUnits(
        df_protocol, directory, 3, read_from
    )
    # Pick the rignt image, thsi should be hard coded in the end

    
    if CCDunit.channel == "NADIR":  # Hack since we dont have any nadir flat fields yet.
        # Cannot be zero due to zero devision in calculate_flatfield. Should be fixed.
        raise Warning('No flatfields measurements of the NADIR channel')
        flatfield = np.zero((511, 2048))+0.01

    else:
        CCDItemsUnitsSelect = list(
            filter(lambda x: (x.imageItem["channel"] == CCDunit.channel), CCDItemsUnits)
        )

        if len(CCDItemsUnitsSelect) > 1:
            print("Several possible pictures found")
        try:
            flatfield = CCDItemsUnitsSelect[
                0
            ].subpic  # This is where it gets read in. The dark (including offsets and balnks) have already been subracted.
        except:
            print("No flatfield CCDItemUnit found - undefined flatfield")

    return flatfield





def scale_field(field):
    # Now scale the average of the middle part of the flatfield with baffle
    # to be the same as the average of flatfield_wo_baffle_lin_scaled .

    # # The values of the below should give an area not affected by the baffle
    FirstRow = 350 #100
    LastRow = 400 #400
    FirstCol =400 #200
    LastCol =1600 #1850

    field_scaled =field/field[FirstRow:LastRow, FirstCol + 1 : LastCol + 1].mean()

    return field_scaled

#%%
    

def read_flatfield_w_baffle(calibration_file, channel):

    calibration_data=toml.load(calibration_file)

    directory = calibration_data["flatfield"][
        "baffle_flatfield"
    ]  #'/Users/lindamegner/MATS/retrieval/Calibration/Final_AIT_2021/LimbFlatfield/Flatfield20210421/PayloadImages/'

    # Read in flatfields taken in April 2021 with baffle Snippet frpm protocol1.txt below
    # TODO: add filename to calibration_file or autoread
    if channel == "IR1":
        filelist = [
            "1303052462743530240_1",
            "1303052550753753600_1",
            "1303052736716888320_1",
            "1303052669289184512_1",
        ]
    elif channel == "IR2":
        filelist = [
            "1303052978547042816_4",
            "1303053042442397952_4",
            "1303053108130874624_4",
            "1303053177900405760_4",
        ]
    elif channel == "IR3":
        filelist = [
            "1303053542163925248_3",
            "1303053614869430528_3",
            "1303053680075119104_3",
            "1303053754744018432_3",
        ]
    elif channel == "IR4":
        filelist = [
            "1303053854620956416_2",
            "1303053921036712704_2",
            "1303053986963058432_2",
            "1303054064460723968_2",
        ]
    elif channel == "UV2":
        filelist = [
            "1303054184923202560_6",
            "1303054301145446656_6",
            "1303054436714050304_6",
            "1303054566244445696_6",
        ]
    elif channel == "UV1":
        filelist = [
            "1303054749741714432_5",
            "1303055407319107072_5",
            "1303055745598846464_5",
            "1303056141568969728_5",
        ]
    elif channel == "NADIR":
        filelist = [""]

    pfile = filelist[1]
    dfile0 = filelist[0]
    dfile2 = filelist[2]


    pic = np.float64(Image.open(directory + pfile + ".pnm"))  # read image
    picd = (np.float64(Image.open(directory + dfile0 + ".pnm"))+
        np.float64(Image.open(directory + dfile2 + ".pnm")))/2. # read dark background


    flatfield_w_baffle = pic - picd

    return flatfield_w_baffle


def read_flatfield_wo_baffle(calibration_file, channel, sigmode='HSM'):
    CCDunit=CCD(channel,calibration_file)
    calibration_data=toml.load(calibration_file)
    flatfield_wo_baffle = read_flatfield(CCDunit, sigmode, calibration_data["flatfield"]["flatfieldfolder_cold_unprocessed"])
 
    return flatfield_wo_baffle


def select_edge_of_baffle_by_plotting(diff_field,zs, plot=True):
    #diff_field should be the ratiofiled between with and without baffle
    #zs is the smoothed field

    x=np.arange(diff_field.shape[1])
    y=np.arange(diff_field.shape[0])
    xx, yy = np.meshgrid(x, y)
    #xx, yy = np.mgrid[np.arange(diff_field.shape[0]), np.arange(diff_field.shape[1])]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, zs, linewidth=2)
        plt.show()

        row=250
        column=1600
        fig = plt.figure()
        ax = fig.subplots(1,2)
        ax[0].plot(diff_field[row,:], color='b')
        ax[0].plot(zs[row,:], color='r')
        ax[0].set_title('fit to row')
        ax[1].plot(diff_field[:,column], color='b')
        ax[1].plot(zs[:,column], color='r')


        fig = plt.figure()
        ax = fig.subplots(1,2)
        ax[0].plot(np.gradient(zs[row,:]), color='r')
        ax[0].set_title('row derivative of sav gloay ')
        ax[0].set_ylim([-0.0005, 0.0025])
        ax[1].plot(np.gradient(zs[:,column]), color='r')
        ax[1].set_title('column derivative of sav gloay ')
        ax[1].set_ylim([-0.0005, 0.0025])




    [ddx,ddy]=np.gradient(zs)
    grad2d=ddx**2+ddy**2
    #ddx[np.isnan(ddx)] = 0.0004
    #ddy[np.isnan(ddy)] = 0.0004


    if plot:
        fig = plt.figure()
        ax = fig.subplots(4,1)
        plot_CCDimage(zs,fig, ax[0], clim=[0.7, 1.1], title='savizky golay fit')
        plot_CCDimage(ddx,fig, ax[1], title='ddx')
        plot_CCDimage(ddy,fig, ax[2], title='ddy')
        
        #grad2d=-grad2dwhere(ddx<0 or ddy<0)
        #grad2d=-grad2d[np.where(ddx < 0 )]
        plot_CCDimage(grad2d,fig, ax[3], title='grad2d')

        


    if plot:
        fig = plt.figure()
        ax = fig.subplots(1)
        grad2plot=plot_CCDimage(grad2d,fig, ax, title='gradient field to make edges visible')
        grad2clim=grad2plot.get_clim()
        fig = plt.figure()
        ax = fig.subplots(2,2)    
        plot_CCDimage(grad2d[450:,0:300],fig, ax[0,0],clim=grad2clim, title='grad2d from row 450 col 0 to 300')
        plot_CCDimage(grad2d[450:,1800:],fig, ax[0,1],clim=grad2clim, title='grad2d from row 450 col 1800 up')
        plot_CCDimage(grad2d[100:300,0:300],fig, ax[1,0],clim=grad2clim, title='grad2d from row 100-300 col 0 to 300')
        plot_CCDimage(grad2d[100:300,1800:],fig, ax[1,1],clim=grad2clim, title='grad2d from row 100-300 col 1800 up')


    return grad2d

def define_edge_of_baffle(channel):
    """ These values are selected using the function select_edge_of_baffle_by_plotting """


    #define area where the particuar ccd is free of baffle contribution. 
    # This is done by eye from the plots of the 2d gradient of the savizky golay fit
    dcol=1820
    drow=280
    if channel=='IR1':
        colstart=30
        colstop=colstart+dcol
        rowstart=200
        rowstop=rowstart+drow
    if channel=='IR2':
        colstart=50
        colstop=colstart+dcol
        rowstart=200
        rowstop=rowstart+drow
    if channel=='IR3': #todi
        colstart=100
        colstop=colstart+dcol
        rowstart=200
        rowstop=rowstart+drow
    if channel=='IR4':
        colstart=190
        colstop=colstart+dcol
        rowstart=170
        rowstop=rowstart+drow
    if channel=='UV1':
        colstart=140
        colstop=colstart+dcol
        rowstart=170
        rowstop=rowstart+drow       
    if channel=='UV2':
        colstart=190
        colstop=colstart+dcol
        rowstart=320
        rowstop=500


    return colstart, colstop, rowstart, rowstop



def scalefieldtoedgevalue(ratiofield, colstart, colstop, rowstart, rowstop, nnpix=2, npix=50):
    """
    #create a field that the without baffle fiels should be divided with to add the baffle effetc
    # this field naturally should be unity in the middle where we trust the without baffle field

    nnpix=2 #number of pixel over which the average brighness at the edge (clear of the baffle)
    npix=50 #range over which to smooth
        #is determined
    
    # ratiofield should be the smoothed ratio 

    """
    [nrow, ncol]=ratiofield.shape    
    scalefield=np.ones(ratiofield.shape)

    



    for irow in range(rowstart, rowstop):
        refcolstart=ratiofield[irow,colstart:colstart+nnpix].mean() # mean value of the pixels just inside the baffle edge
        scalefield[irow, 0:colstart]=ratiofield[irow, 0:colstart]/refcolstart
        for i in range(0, npix+1): #linearly scale up so that the value at the exact edge matches
            scalefield[irow, colstart+i]=(npix-i)/npix*scalefield[irow, colstart-1]+i/npix*1. # lineary scale so that after  npix the calue is unity
        refcolstop=ratiofield[irow,colstop-nnpix:colstop].mean() # mean value of the pixels just inside the baffle edge
        scalefield[irow, colstop:ncol]=ratiofield[irow, colstop:ncol]/refcolstop
        for i in range(0, npix+1): #linearly scale up so that the value at the exact edge matches
            scalefield[irow, colstop-i]=(npix-i)/npix*scalefield[irow, colstop]+i/npix*1. # lineary scale so that after  npix the calue is unity


    for icol in range(colstart, colstop):
        refrowstart=ratiofield[rowstart:rowstart+nnpix, icol].mean()
        scalefield[0:rowstart, icol]=ratiofield[0:rowstart, icol]/refrowstart
        for i in range(0, npix+1): #linearly scale up so that the value at the exact edge matches
            scalefield[rowstart+i, icol]=(npix-i)/npix*scalefield[rowstart-1,icol]+i/npix*scalefield[rowstart-1+npix,icol] # lineary scale so that after  npix the calue is unity
        if rowstop<nrow:
            refrowstop=ratiofield[rowstop-nnpix:rowstop, icol].mean()
            scalefield[rowstop:nrow, icol]=ratiofield[rowstop:nrow, icol]/refrowstop
            for i in range(0, npix+1): #linearly scale up so that the value at the exact edge matches
                scalefield[rowstop-i, icol]=(npix-i)/npix*scalefield[rowstop,icol]+i/npix*scalefield[rowstop-npix,icol] # lineary scale so that after  npix the calue is unity
    
 #now fill the corners by multiplying the edges values
    for irow in range(0, rowstart):
        refrowstartcolstart=ratiofield[rowstart:rowstart+nnpix, colstart:colstart+nnpix].mean()
        for icol in range(0,colstart):
            scalefield[irow, icol]=ratiofield[irow, icol]/refrowstartcolstart
        refrowstartcolstop=ratiofield[rowstart:rowstart+nnpix, colstop-nnpix:colstop].mean()
        for icol in range(colstop, ncol):
            scalefield[irow, icol]=ratiofield[irow, icol]/refrowstartcolstop

    for irow in range(rowstop, nrow):
        refrowstopcolstart=ratiofield[rowstop-nnpix:rowstop, colstart:colstart+nnpix].mean()
        for icol in range(0,colstart):
            scalefield[irow, icol]=ratiofield[irow, icol]/refrowstopcolstart
        refrowstopcolstop=ratiofield[rowstop-nnpix:rowstop, colstop-nnpix:colstop].mean()
        for icol in range(colstop, ncol):
            scalefield[irow, icol]=ratiofield[irow, icol]/refrowstopcolstop

    return scalefield


#%%

def make_flatfield(channel, calibration_file, plotresult=False, plotallplots=False):
    import mats_l1_processing
    from sgolay2 import SGolayFilter2

    #Note the high signal mode flatfields are used even if in low signal mode. They are scaled to be 1 in the middle of the field anyway.
    flatfield_wo_baffle=read_flatfield_wo_baffle(calibration_file, channel, sigmode='HSM')
    flatfield_w_baffle=read_flatfield_w_baffle(calibration_file, channel)
   

    flatfield_wo_baffle_scaled=scale_field(flatfield_wo_baffle)
    flatfield_w_baffle_scaled=scale_field(flatfield_w_baffle)
    ratio_w_to_wo_scaled=flatfield_w_baffle_scaled/flatfield_wo_baffle_scaled
    zs = SGolayFilter2(window_size=31, poly_order=1)(ratio_w_to_wo_scaled)
    grad2d=select_edge_of_baffle_by_plotting(ratio_w_to_wo_scaled,zs, plot=plotallplots)




    colstart, colstop, rowstart, rowstop=define_edge_of_baffle(channel) #Note: a manual selection is needed based on select_edge_of_baffle_by_plotting
    scalefield=scalefieldtoedgevalue(zs, colstart, colstop, rowstart, rowstop, nnpix=2, npix=50)




    if plotresult:
        fig, ax = plt.subplots(4,2, figsize=[8,10])

        plot_CCDimage(flatfield_wo_baffle_scaled,fig, ax[0,0], title=channel+' wo flatfield scaled')
        plot_CCDimage(flatfield_w_baffle_scaled,fig, ax[0,1], title=channel+ 'w flatfield scaled')

        plot_CCDimage(ratio_w_to_wo_scaled,fig, ax[1,0], title=channel+' ratio btw scaled w and scaled wo ')
        plot_CCDimage(zs,fig, ax[1,1], clim=[0.7, 1.1], title=channel+' sav golay fit')

        plot_CCDimage(scalefield,fig, ax[2,0], clim=[0.7, 1.1], title=channel+' scalefield')
        myflatfield=scalefield*flatfield_wo_baffle_scaled
        plot_CCDimage(myflatfield,fig, ax[2,1], title=channel+' merged flatfield')
        plot_CCDimage(flatfield_wo_baffle_scaled-myflatfield,fig, ax[3,0], title='difference wo flatfield -  merged flatfield')
        plot_CCDimage(flatfield_w_baffle_scaled-myflatfield,fig, ax[3,1], title='difference w flatfield - merged flatfield')

        plt.tight_layout()
        Path("output").mkdir(parents=True, exist_ok=True)
        fig.savefig("output/Merged_Flatfield_" + channel + ".jpg")


    return myflatfield



# %%
