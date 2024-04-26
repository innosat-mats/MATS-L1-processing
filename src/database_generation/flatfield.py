#%%

import numpy as np
from PIL import Image
from mats_l1_processing.instrument import CCD
from database_generation.experimental_utils import plot_CCDimage
import matplotlib.pyplot as plt

from pathlib import Path
import toml



def read_flatfield(CCDunit, mode, flatfield_directory, reporterror=False):
    from mats_l1_processing.items_units_functions import (
        read_files_in_protocol_as_ItemsUnits,
    )
    from database_generation.experimental_utils import readprotocol

    if mode == 'HSM': 
        directory = flatfield_directory
        protocol='flatfields_200330_SigMod1_LMprotocol.txt'
        #protocol = "readin_flatfields_SigMod1.txt"
        #directory = '/Users/lindamegner/MATS/retrieval/Calibration/AfterLightLeakage/Flatfields/20200429_flatfields_8C/'
        #protocol='protocolx2.txt'

    elif mode == 'LSM':  # LSM Should not be used anymore. Only HSM
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
        raise Warning('No flatfields measurements of the NADIR channel')
        flatfield = np.ones((511, 2048))

    else:
        CCDItemsUnitsSelect = list(
            filter(lambda x: (x.imageItem["channel"] == CCDunit.channel), CCDItemsUnits)
        )

        if len(CCDItemsUnitsSelect) < 1:
            print("No flatfield CCDItemUnit found - undefined flatfield")
        

        #Set signal limits for when the flatfield is used since we dont want only dark current and we dont want saturation
        if CCDunit.channel in ["IR1", "IR2", "IR3", "IR4"]:
            signalmin=600
            signalmax=2800
        elif CCDunit.channel in ["UV1", "UV2"]:#lower signal must be acceptes in UV since the signal is lower
            signalmin=200
            signalmax=2800
        flatfieldlist = []  # Define flatfieldlist
        for i in range(1, len(CCDItemsUnitsSelect)): #the first image is not used since it is weird for unknown reasons
            flatfieldtmp = CCDItemsUnitsSelect[i].subpic
            if flatfieldtmp.mean()>signalmin and flatfieldtmp.mean()<signalmax:
                #flatfieldlist.append(CCDItemsUnitsSelect[i].subpic/(CCDItemsUnitsSelect[i].imageItem["TEXPMS"]/1000.))  # Append to flatfieldlist
                flatfieldlist.append(scale_field(flatfieldtmp)) # Append flatfield scaled to unity in the middle to flatfieldlist
                #print('i=',i,'channel:', CCDItemsUnitsSelect[i].imageItem["channel"])
                #plot_CCDimage(CCDItemsUnitsSelect[i].subpic, title='flatfieldlist[i]'+str(i)+ 'texpms'+str(CCDItemsUnitsSelect[i].imageItem["TEXPMS"]/1000.))
 
        try:
            flatfield = np.array(flatfieldlist).mean(axis=0)
            flatfielderr = np.array(flatfieldlist).std(axis=0)/np.sqrt(len(flatfieldlist))
            
        except:
            Exception("No flatfield CCDItemUnit found - undefined flatfield")
            
        #plot_CCDimage(flatfield, title='flatfield')
        #plot_CCDimage(flatfielderr, title='flatfieldstd/sqrt(n)')

    if reporterror:
        return flatfield, flatfielderr
    else:
        return flatfield






def scale_field(field, errorfield=None):
    # Now scale the average of the middle part of the flatfield with baffle
    # to be unity.

    # # The values of the below should give an area not affected by the baffle and is the area that the 
    # calibration factors from the lab is defined by

    FirstRow = 350 
    LastRow = 400 
    FirstCol =524
    LastCol =1523

    field_scaled =field/field[FirstRow:LastRow, FirstCol + 1 : LastCol + 1].mean()
    if errorfield is not None:
        errorfield_scaled =errorfield/field[FirstRow:LastRow, FirstCol + 1 : LastCol + 1].mean()
        return field_scaled, errorfield_scaled
    else:
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




    
def read_second_flatfield_w_baffle(calibration_file, channel):

    # Read in a flatfield taken in a different way to compare with and use for error estimation
    # For IR1, IR2, IR4 and UV1 and UV2 a vertical flatfield is used.
    #For IR3 another horizontal faltfield is used since the vertical was not good
    # Reading in images without racfiles.
    calibration_data=toml.load(calibration_file)
    if channel != "IR3":
        directory = calibration_data["flatfield"]["baffle_flatfield_secondIR124UV12"]
        if channel == "IR1":
            filelist = [
                "1315139160698745856_1",
                "1315139215614486784_1",
                "1315140683491622912_1",
                "1315140750606918400_1"
            ]
        elif channel == "IR2":
            filelist = [
                "1315139318878006016_4",
                "1315139365331344640_4",
                "1315140593069412352_4",
                "1315140637277480960_4",
            ]

        elif channel == "IR4":
            filelist = [
                "1315139415534683136_2",
                "1315139462421523968_2",
                "1315140510849411072_2",
                "1315140558911407360_2",
            ]

        elif channel == "UV2":
            filelist = [
                "1315139512226760960_6",
                "1315139614537979136_6",
                "1315140345809494016_6",
                "1315140468414031872_6",
            ]

        elif channel == "UV1":
            filelist = [
                "1315139654592239360_5",
                "1315139943497085696_5",
                "1315140020567672832_5",
                "1315140303270797824_5",
            ]

        elif channel == "NADIR":
            filelist = [""]


    else: # if channel == "IR3":
        # Reading in the horozizonal flatfield for IR3 since the vertical flatfield was not good.
        # Also reading from another directory since IR# was not functioning at the first test camapign

        directory = calibration_data["flatfield"]["baffle_flatfield_secondIR3"]

        filelist = [
            "1315316689863510016_3",
            "1315316743288696320_3",
            "1315316834330658048_3",
            "1315316902079223552_3",
        ]


    pfile_B = filelist[0]
    pfile_SB = filelist[1]
    pfile_D = filelist[2]
    pfile_SD = filelist[3]

    pic_B = np.float64(Image.open(directory + pfile_B + ".pnm"))
    pic_SB = np.float64(Image.open(directory + pfile_SB + ".pnm"))
    pic_D = np.float64(Image.open(directory + pfile_D + ".pnm"))
    pic_SD = np.float64(Image.open(directory + pfile_SD + ".pnm"))
    flatfield_w_baffle = pic_B - pic_SB - (pic_D - pic_SD)

    return flatfield_w_baffle



def read_flatfield_wo_baffle(calibration_file, channel, sigmode='HSM', reporterror=False):
    CCDunit=CCD(channel,calibration_file)
    calibration_data=toml.load(calibration_file)
    flatfield_wo_baffle, flatfield_wo_baffle_err = read_flatfield(CCDunit, sigmode, calibration_data["flatfield"]["flatfieldfolder_cold_unprocessed"], reporterror=True)
    if reporterror:
        return flatfield_wo_baffle, flatfield_wo_baffle_err
    else:
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
    #Readis in flatfields with baffle taken from the lab at OHB. And reads in another flatfield with baffle 
    # so that the difference between the two can be used to estimate the error. The second flatfield is 
    # vertically for IR1, IR2 and IR4. For IR3 there was something in hte image when the vertical field was measured so 
    # another horizontal field is used. For UV1 and UV2 the flatfields errors....what to do here
    


    from scipy.signal import medfilt

    #Note the high signal mode flatfields are used even if in low signal mode. 
    rawflatfield=read_flatfield_w_baffle(calibration_file, channel)
    

    # Read in a flatfield taken in a different way to compare with and use for error estimation
    # For IR1, IR2, IR4 and UV1 and UV2 a vertical flatfield is used.
    #For IR3 another horizontal faltfield is used since the vertical was not good

    rawflatfield2=read_second_flatfield_w_baffle(calibration_file,channel)

    # Remove hot pixels by applying a median filter to every row in the 2D array, and rescale. 
    flatfield_smooth = np.apply_along_axis(lambda x: medfilt(x, kernel_size=3), axis=1, arr=rawflatfield)  
    flatfield2_smooth = np.apply_along_axis(lambda x: medfilt(x, kernel_size=3), axis=1, arr=rawflatfield2)

    
    flatfield_scaled=scale_field(flatfield_smooth)
    flatfield2_scaled=scale_field(flatfield2_smooth)
    flatfield_err=(flatfield2_scaled-flatfield_scaled)




         
    



    


    return flatfield_scaled, flatfield_err



# %%
