#%%
import mats_l1_processing
import numpy as np
from PIL import Image
from mats_l1_processing.instrument import CCD
from mats_l1_processing.experimental_utils import (
    plot_CCDimage,
    read_all_files_in_protocol,
)
from mats_l1_processing.experimental_utils import readprotocol
import matplotlib.pyplot as plt

# from scipy import signal
from scipy import ndimage, io
from scipy.signal import spline_filter
from pathlib import Path
import toml



def read_flatfield(CCDunit, mode, flatfield_directory):
    from mats_l1_processing.items_units_functions import (
        read_files_in_protocol_as_ItemsUnits,
    )
    from mats_l1_processing.experimental_utils import readprotocol

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



def coef_for_uneven_lighting(image, hmin, hmax, vmin, vmax, ax=None):

    #Correct for slant due to uneven lighting of screen
    x=np.arange(hmin,hmax)

    meanmean=image[vmin:vmax,hmin:hmax].mean()
    mean_v=image[vmin:vmax,hmin:hmax].mean(0)/meanmean

    coef = np.polyfit(x, mean_v, 1) 

    if ax: #plotting for analysis
        #poly1d_fn = np.poly1d(coef)
        #ax.plot(x,mean_v, x, poly1d_fn(x))
        ax.plot(x,mean_v)
    return coef

def correct_for_uneven_lighting(image, coef):
    #Correcting for the uneven lighting by using coefficients given by  coef_for_uneven_lighting
    #plt.plot(mean_v) 
    poly1d_fn = np.poly1d(coef)
    #plt.plot(x,mean_v, 'yo', x, poly1d_fn(x), '--k')

    #print(coef)
    image_lightcorr=image.copy()
    for icol in np.arange(0, image.shape[1]): #all columns
        image_lightcorr[:,icol]=image[:,icol]/poly1d_fn(icol)  
    
    return image_lightcorr



def mask_baffle_interference(diff_field,npix, threshold):
    """
    fuction that returns a mask with true when there is no baffle intereence and false where it is close to the baffle

    Input:
    diff_field: Numpy array


    Output:
    no_baffle_interference: boolean

    """ 

    no_baffle_interference= np.ones(diff_field.shape)

    midrow=int(diff_field.shape[0]/2)
    maxrow=diff_field.shape[0]
    midcol=int(diff_field.shape[1]/2)
    maxcol=diff_field.shape[1]
    for irow in range(0, maxrow):
        row=diff_field[irow,:] 
        mask_row=no_baffle_interference[irow,:]
        for icol in range(1, midcol):#fix "left" side first
            istart=icol-npix
            if (istart<0): 
                istart=0 #make sure it is not negative
            if row[istart:icol].mean()<threshold: # if the image is darkend
                mask_row[icol]=0 # Meaning that there is baffle interference
        for icol in range(midcol,maxcol): #fix "right" side
            istop=icol+npix
            if istop>len(row)+1: istop=len(row)+1 #maximem lenght of row
            if row[icol:istop].mean()<threshold: # if the image is darkend
                mask_row[icol]=0 # Meaning that there is baffle interference

    for icol in range(0, maxcol):
        col=diff_field[:,icol]    
        mask_col=no_baffle_interference[:,icol]     
        for irow in range(1,midrow):#lower half of image
            istart=irow-npix
            if istart<0: istart=0 #make sure it is not negative
            if col[istart:irow].mean()<threshold: # if the image is darkend
                mask_col[irow]=0 # Meaning that there is baffle interference    
        for irow in range(midrow, maxrow):#upperr half of image
            istop=irow+npix
            if istop>len(col)+1: istop=len(row)+1 #make sure it is not negative
            if col[irow:istop].mean()<threshold: # if the image is darkend
                mask_col[irow]=0 # Meaning that there is baffle interference 
     
    return no_baffle_interference


def scale_field(field):
    # Now scale the average of the middle part of the flatfield with baffle
    # to be the same as the average of flatfield_wo_baffle_lin_scaled .

    # # The values of the below should give an area not affected by the baffle
    FirstRow = 300 #100
    LastRow = 400 #400
    FirstCol =400 #200
    LastCol =1600 #1850

    field_scaled =field/field[FirstRow:LastRow, FirstCol + 1 : LastCol + 1].mean()

    return field_scaled

#%%
def make_flatfield_lin_coef(channel, signalmode, calibration_file, ax=None):

    # creates coefficients for a linear correction due to uneven lighting. Copied from the 
    #start of make_flatfield but needed in separate function so that flatfield from one channel
    #can be used for all

    CCDunit=CCD(channel,calibration_file)

    calibration_data=toml.load(calibration_file)

    flatfield_wo_baffle = read_flatfield(CCDunit, signalmode, calibration_data["flatfield"]["flatfieldfolder_cold_unprocessed"])


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
    #dfile = filelist[2]

    pic = np.float64(Image.open(directory + pfile + ".pnm"))  # read image
    picd = (np.float64(Image.open(directory + dfile0 + ".pnm"))+
        np.float64(Image.open(directory + dfile2 + ".pnm")))/2. # read dark background
    #picd = np.float64(Image.open(directory + dfile + ".pnm"))


    flatfield_w_baffle = pic - picd


    #Flip fields for flipped channels:
    if (channel=='IR1' or channel=='IR3'or channel=='UV1' or channel=='UV2'):
        flatfield_wo_baffle=np.fliplr(flatfield_wo_baffle)
        flatfield_w_baffle=np.fliplr(flatfield_w_baffle)

    flatfield_wo_baffle_scaled=scale_field(flatfield_wo_baffle)
    flatfield_w_baffle_scaled=scale_field(flatfield_w_baffle)

    #Correct for lamp distance to angled screen - more light on one side than the other
    # Set limit to perform linear fit across the with of the CCD to correct for this 
    area_ymin = 300
    area_ymax = 400
    area_xmin= 400
    area_xmax= 1600    

    coef_wo=coef_for_uneven_lighting(flatfield_wo_baffle_scaled,area_xmin,area_xmax, area_ymin, area_ymax, ax=ax)
    coef_w=coef_for_uneven_lighting(flatfield_w_baffle_scaled,area_xmin,area_xmax, area_ymin, area_ymax, ax=ax)
    
    return coef_wo, coef_w




#%%
def make_flatfield(channel, signalmode, calibration_file, coef_wo=None, coef_w=None,plot=True):

    # makes flatfield using both a cold flatfield without baffle and a room temp flatfield with baffle.



    CCDunit=CCD(channel,calibration_file)

    calibration_data=toml.load(calibration_file)

    flatfield_wo_baffle = read_flatfield(CCDunit, signalmode, calibration_data["flatfield"]["flatfieldfolder_cold_unprocessed"])


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


    #Flip fields for flipped channels:
    if (channel=='IR1' or channel=='IR3'or channel=='UV1' or channel=='UV2'):
        flatfield_wo_baffle=np.fliplr(flatfield_wo_baffle)
        flatfield_w_baffle=np.fliplr(flatfield_w_baffle)

    flatfield_wo_baffle_scaled=scale_field(flatfield_wo_baffle)
    flatfield_w_baffle_scaled=scale_field(flatfield_w_baffle)


    # #Correct for lamp distance to angled screen - more light on one side than the other

    if ((coef_wo is None) or (coef_w is None)):
        # Set limit to perform linear fit across the with of the CCD to correct for this 
        area_ymin = 300
        area_ymax = 400
        area_xmin= 400
        area_xmax= 1600    

        coef_wo_p=coef_for_uneven_lighting(flatfield_wo_baffle_scaled,area_xmin,area_xmax, area_ymin, area_ymax)
        coef_w_p=coef_for_uneven_lighting(flatfield_w_baffle_scaled,area_xmin,area_xmax, area_ymin, area_ymax)

        flatfield_wo_baffle_lin_scaled=correct_for_uneven_lighting(flatfield_wo_baffle_scaled,coef_wo_p)
        flatfield_w_baffle_lin_scaled=correct_for_uneven_lighting(flatfield_w_baffle_scaled,coef_w_p)
    else:   
        flatfield_wo_baffle_lin_scaled=correct_for_uneven_lighting(flatfield_wo_baffle_scaled,coef_wo)
        flatfield_w_baffle_lin_scaled=correct_for_uneven_lighting(flatfield_w_baffle_scaled,coef_w)

    #flatfield_wo_baffle_lin_scaled = flatfield_wo_baffle_lin / np.mean(flatfield_wo_baffle_lin[area_ymin:area_ymax, area_xmin + 1 : area_xmax + 1])



    diff_field = flatfield_w_baffle_lin_scaled-flatfield_wo_baffle_lin_scaled 
    # diff_field[FirstRow:LastRow, FirstCol + 1 : LastCol + 1]=0.

    diff_field = ndimage.gaussian_filter(diff_field, sigma=10)
    no_baffle_interference= mask_baffle_interference(diff_field, npix=10, threshold=-0.05)
    #diff_field[np.where((abs(diff_field)<0.5) & (no_baffle_interference==1))]=0.
    diff_field[np.where(no_baffle_interference==1)]=0.

   



    #diff_field[abs(diff_field)<0.02]&no_baffle_interference=0
    diff_field_smooth = ndimage.gaussian_filter(diff_field, sigma=30)  # sigma = 5 pixels for gaussan smoothing kernel?

    # add baffle effects to flatfield without baffle except where the baffle has no impact (TBD)
    flatfield_morphed = flatfield_wo_baffle_lin_scaled+diff_field_smooth
    # flatfield_morphed[
    #     FirstRow:LastRow, FirstCol + 1 : LastCol + 1
    # ] = flatfield_wo_baffle_lin_scaled[FirstRow:LastRow, FirstCol + 1 : LastCol + 1]

    #
    flatfield_morphed_minus_wo_scaled = flatfield_morphed - flatfield_wo_baffle_lin_scaled
    flatfield_morphed_minus_w_scaled = flatfield_morphed - flatfield_w_baffle_lin_scaled

    if plot:
        fig, ax = plt.subplots(4,2, figsize=(10,10))
        # Plotting limits
        fullpic = True
        if fullpic:
            xpmin = 0
            xpmax = flatfield_w_baffle.shape[1]
            ypmin = 0
            ypmax = flatfield_w_baffle.shape[0]
        else:
            xpmin = 1000
            xpmax = 1100
            ypmin = 300
            ypmax = 400

        # plot_CCDimage(
        #     flatfield_wo_baffle[ypmin:ypmax, xpmin:xpmax],
        #     fig,
        #     ax[0,0],
        #     title="flatfield_wo_baffle",
        # )
        # plot_CCDimage(
        #     flatfield_w_baffle[ypmin:ypmax, xpmin:xpmax],
        #     fig,
        #     ax[0,1],
        #     title="flatfield_w_baffle",
        #)

        sp_scaled1=plot_CCDimage(
            flatfield_wo_baffle_scaled[ypmin:ypmax, xpmin:xpmax],
            fig,
            ax[0,0],
            title=channel+"flatfield_wo_baffle_scaled",
        )
        clim1=sp_scaled1.get_clim()
        plot_CCDimage(
            flatfield_w_baffle_scaled[ypmin:ypmax, xpmin:xpmax],
            fig,
            ax[0,1],
            clim=clim1,
            title=channel+"flatfield_w_baffle_scaled",
        )

        # plot_CCDimage(
        #     flatfield_wo_baffle_scaled[ypmin:ypmax, xpmin:xpmax]-flatfield_w_baffle_scaled[ypmin:ypmax, xpmin:xpmax],
        #     fig,
        #     ax[5,1],
        #     title=channel+"flatfield_wo_baffle_scaled-flatfield_w_baffle_scaled",
        # )








        sp_scaled=plot_CCDimage(
            flatfield_wo_baffle_lin_scaled[ypmin:ypmax, xpmin:xpmax],
            fig,
            ax[1,0],
            title="flatfield_wo_baffle_lin_scaled",
        )

        clim=sp_scaled.get_clim()
        plot_CCDimage(
            flatfield_w_baffle_lin_scaled[ypmin:ypmax, xpmin:xpmax],
            fig,
            ax[1,1],
            clim=clim,
            title="flatfield_w_baffle_lin_scaled",
        )
        
        plot_CCDimage(
            flatfield_morphed[ypmin:ypmax, xpmin:xpmax],
            fig,
            ax[2,0],
            clim=clim,
            title="morphed flatfield",
        )

        plot_CCDimage(
            flatfield_morphed[ypmin:ypmax, xpmin:xpmax],
            fig,
            ax[2,1],
            clim=clim,
            title="morphed flatfield",
        )


        sp_diff2=plot_CCDimage(
            flatfield_morphed_minus_w_scaled[ypmin:ypmax, xpmin:xpmax],
            fig,
            ax[3,1],
            #clim=[-0.4, 0],
            title="flatfield_morphed-flatfield_w_baffle_lin_scaled",
        )
        
        clim_diff2=sp_diff2.get_clim()
        plot_CCDimage(
            flatfield_morphed_minus_wo_scaled[ypmin:ypmax, xpmin:xpmax],
            fig,
            ax[3,0],
            clim=clim_diff2,
            title="flatfield_morphed-flatfield_wo_baffle_lin_scaled",
        )
 
        # sp_diff_field=plot_CCDimage(
        #     diff_field[ypmin:ypmax, xpmin:xpmax],
        #     fig,
        #     ax[4,0],
        #     clim=[-0.05, 0.05],
        #     title="difference in scaled flatf without and with baffle",
        # )
        # clim_diff=sp_diff_field.get_clim()

        # # diff_field_smooth= spline_filter(diff_field, lmbda=4.)
        # plot_CCDimage(
        #     diff_field_smooth[ypmin:ypmax, xpmin:xpmax],
        #     fig,
        #     ax[4,1],
        #     clim=clim_diff,
        #     #clim=[-0.1, 0.1],
        #     title="smooth difference in scaled flatf without and with baffle",
        # )



        # for myax in ax:

        #    myax.set_ylim((300,400))
        #    myax.set_xlim((1000,1100))
        #    myax.set_aspect('auto')

        fig.suptitle(channel+' '+signalmode)

        plt.tight_layout()
        Path("output").mkdir(parents=True, exist_ok=True)
        fig.savefig("output/MorphedFlatfield_" + channel + ".jpg")


    # UnFlip fields for flipped channels:
    if (channel=='IR1' or channel=='IR3'or channel=='UV1' or channel=='UV2'):
        flatfield_morphed=np.fliplr(flatfield_morphed)



    return flatfield_morphed
