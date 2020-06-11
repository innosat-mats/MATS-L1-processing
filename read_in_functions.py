#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 08:57:37 2020

@author: lindamegner

Functions used to read in MATS images and data in different ways: From KTH, from Immage viewer and from rac files. 

The housekeepting data temperatures can also be read in using these functions.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import json
from PIL import Image



import imagereader


def plot_full_temperature_info(temperaturedata,relativetimedata):
    HTR1A=temperaturedata[:,0]
    HTR1B=temperaturedata[:,1]
    HTR2A=temperaturedata[:,2]
    HTR2B=temperaturedata[:,3]
    HTR8A=temperaturedata[:,4]
    HTR8B=temperaturedata[:,5]

    plt.plot(relativetimedata/60.,HTR1A,label='splitter plate, regulation')
    plt.plot(relativetimedata/60.,HTR1B,label='splitter plate, measuring')
    plt.plot(relativetimedata/60.,HTR2A,label='limb house, regulation')
    plt.plot(relativetimedata/60.,HTR2B,label='limb house, measuring')
    plt.plot(relativetimedata/60.,HTR8A,label='UV2 CCDn')
    plt.plot(relativetimedata/60.,HTR8B,label='UV1 CCDn')
    plt.xlabel('Time since start of instrument [min]')
    plt.ylabel('Temperature [C]')
    plt.legend()
    plt.show()
    plt.savefig('HTRmeasurements.jpg')

def add_temperature_info_to_CCDitems(CCDitems,read_from,directory,labtemp=999):
    from get_temperature import create_temperature_info_array, add_temperature_info

    
    if read_from=='rac':    
        temperaturedata, relativetimedata=create_temperature_info_array(directory+'RacFiles_out/HTR.csv')
    elif read_from!='rac':
        temperaturedata=999
        relativetimedata=999
    
    #plot_full_temperature_info(temperaturedata,relativetimedata)

    
    for CCDitem in CCDitems:
        CCDitem=add_temperature_info(CCDitem,temperaturedata,relativetimedata,labtemp)
#        timestamp=epoch+datetime.timedelta(0,CCDitem['reltime'])

    return CCDitems


def read_all_files_in_protocol(df,read_from, directory):
    

        
    if read_from=='rac':
        CCDitemsunsorted=read_CCDitems(directory+'RacFiles_out/')
        CCDitems=[]
        for PicID in list(df['PicID']):        
            item=searchlist(CCDitemsunsorted, key='id', value=PicID)
            if item:  #checks that item is not a NoneType object
                CCDitems.append(item)
            else:
                print('Warning: no image file corresponding to an entry in the protocol.')

            
    elif read_from=='imgview':
        CCDitems=readselectedimageviewpics(directory+'PayloadImages/',list(df['PicID']))
    else: 
        raise Exception('read_from must be rac or imgview')
        
    for CCDitem in CCDitems: 

        CCDitem['DarkBright']=df.DarkBright[df.PicID==CCDitem['id']].iloc[0]

      
    return CCDitems    


def readprotocol(filename):
    import pandas as pd
    df = pd.read_csv(filename, sep=" ", comment='#', skipinitialspace=True, skiprows=())
    return df


def searchlist(list, key, value): 
    found=False
    for item in list: 
        if item[key] == value:
            found=True
            return item 
        
    if not found:
        print('Warning: Item not found')

def readimageviewpic(dirname,picnr,rawflag):
# This function should not be needed anymore.          
    if rawflag==1:
        imagefile= dirname +'rawoutput'+str(picnr) +'.pnm'
    else:
        imagefile= dirname +'output'+str(picnr) +'.pnm'
    txtfile=dirname +'output'+str(picnr) +'.txt'
    image_raw = np.float64(Image.open(imagefile))
    CCDitem=read_txtfile_create_CCDitem(txtfile)
    CCDitem['IMAGE']=image_raw    
    return CCDitem


def read_CCDitem_from_imgview(dirname,IDstring):
# This function used to be called readimageviewpic2    
# Almost equivalent to the fucntion read_pnm_image_and_txt but kept since differnt files need different input.      
    imagefile= dirname + IDstring +'.pnm'
    txtfile= dirname + IDstring +'_output.txt'
    try:
        image_raw = np.float64(Image.open(imagefile))
        #image_raw = Image.open(imagefile)
        CCDitem=read_txtfile_create_CCDitem(txtfile)
        CCDitem['IMAGE']=image_raw    
    except:
        print('There is something wrong with image file ',imagefile)
        CCDitem=-999
        raise Exception()
            
    CCDitem['read_from']='imgview'
    try:
        CCDitem['reltime']=1.e-9*CCDitem['EXP Nanoseconds']
    except:
        try:
            CCDitem['reltime']=int(CCDitem['EXPTS'])+int(CCDitem['EXPTSS'])/2**16 
        except:
            raise Exception('No info on the relative time')
    

    return CCDitem
    
def read_pnm_image_and_txt(dirname,picid):
# Reads data from output image file and combines it with the txt file.
    imagefile= dirname +picid +'.pnm'
    txtfile=dirname +picid+'_output.txt'
    image_raw = np.float64(Image.open(imagefile))
    CCDitem=read_txtfile_create_CCDitem(txtfile)
    CCDitem['IMAGE']=image_raw    
    return CCDitem
  

def readimageviewpics(dirname,rawflag=0, filelist=[]):
    from os import listdir
    from os.path import isfile, join    
    #Reads  all images in directory
    all_files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
    pnm_files = list(filter(lambda x: x[-4:] == '.pnm', all_files))
    CCDitems=[]
    for pnm_file in pnm_files:
        IDstring=pnm_file[:-4]
        CCDitem=read_CCDitem_from_imgview(dirname,IDstring,rawflag)
        if CCDitem != -999:
            CCDitems.append(CCDitem)
        
    return CCDitems


def readselectedimageviewpics(dirname,IDlist):
    CCDitems=[]
    for IDstring in IDlist:
        CCDitem=read_CCDitem_from_imgview(dirname,IDstring)
        
        
        if CCDitem != -999:
            CCDitems.append(CCDitem)
            
    return CCDitems






def read_MATS_image(rac_dir):
    import pandas as pd
    from PIL import Image

    # If you use an newer version of the rac extract reader then pathdir is not needed. Nwwer is approximately from the start of 2020. 
#    json_file = open(filename,'r')
#    CCD_image_data = json.load(json_file)
#    json_file.close

    df = pd.read_csv(rac_dir+'CCD.csv', skiprows=[0]) 
    CCD_image_data=df.to_dict('records')

    
    
    for item in CCD_image_data:
#        print(pathdir+str(CCD_image_data[i]['IMAGEFILE']) + '_data.npy')
        pngfile=rac_dir+str(item['Image File Name'])
        jsonfile=pngfile[0:-4]+'.json'
        try:
            item['IMAGE'] = np.float64(Image. open(pngfile))
            with open(jsonfile) as f:
                item['jsondata'] = json.load(f)
       
        except:    
            print('Warning, one image file seems corrupt and has been rmeoved')
            CCD_image_data.remove(item)
    
    return CCD_image_data





def read_CCDitem(rac_dir,PicID, labtemp=999):
# reads data from one image (itemnumber) in the rac file
    from math import log
    import pandas as pd
    from PIL import Image
    from get_temperature import create_temperature_info_array, add_temperature_info

    df = pd.read_csv(rac_dir+'CCD.csv', skiprows=[0]) 
    CCD_image_data=df.to_dict('records')

    
 #   CCD_image_data=read_MATS_image(rac_dir)
 #   CCD_image_data=read_MATS_image(rac_image_json_dir+rac_sub_dir+rac_image_json_file,rac_image_json_dir)

    if PicID.count('_')==1: # new way of naming as of June 2020 in protocol
        itemnumber=int(PicID[:-2])
        CCDSEL=int(PicID[-1:])
        CCDitem=next(item for item in CCD_image_data if item['EXP Nanoseconds'] == itemnumber and item['CCDSEL']== CCDSEL)
    elif PicID.count('_')==2: # old way of naming as of spring 2020 in protociol
        itemnumber=int(PicID.split('_')[0])
        CCDSEL=int(PicID[-1:])
        

        CCDitem=next(item for item in CCD_image_data if str(item['EXP Nanoseconds'])[:-9] == str(itemnumber) and item['CCDSEL']== CCDSEL)
    else:   
        raise Exception('strange naming in protocol, PicID=', PicID)
   # CCDitem=list(filter(lambda item: item['EXP Nanoseconds'] == itemnumber, CCD_image_data))
 

   
    if int(CCDitem['CCDSEL'])==1: #input CCDSEL=1
        channel='IR1'
    elif int(CCDitem['CCDSEL'])==4: #input CCDSEL=8
        channel='IR2'
    elif int(CCDitem['CCDSEL'])==3: #input CCDSEL=4
        channel='IR3'
    elif int(CCDitem['CCDSEL'])==2: #input CCDSEL=2
        channel='IR4'
    elif int(CCDitem['CCDSEL'])==5: #input CCDSEL=16
        channel='UV1'
    elif int(CCDitem['CCDSEL'])==6: #input CCDSEL=32
        channel='UV2'
    elif int(CCDitem['CCDSEL'])==7: #input CCDSEL=64
        channel='NADIR'
    else:
        print('Error in CCDSEL, CCDSEL=',int(CCDitem['CCDSEL']))  
        
        
        
                
    CCDitem['channel']=channel
#   Renaming of stuff. The names in the code here is based on the old rac extract file (prior to May 2020) rac_extract file works        
    CCDitem['id']=str(CCDitem['EXP Nanoseconds'])+'_'+str(CCDitem['CCDSEL'])
        
  # TODO LM June 2020: Change  all code so that the new names, i. CCDitem['NCBIN CCDColumns'] and CCDitem['NCBIN FPGAColumns'] are used instead of the old.
    try:
        CCDitem['NColBinCCD']
    except:
        CCDitem['NColBinCCD']=CCDitem['NCBIN CCDColumns']
   
    #CCDitem['NColBinFPGA']=CCDitem['NCBIN FPGAColumns']
    try:  
        CCDitem['NColBinFPGA']
    except: 
        CCDitem['NColBinFPGA']=log(CCDitem['NCBIN FPGAColumns'])/log(2)
       
        #del CCDitem['NCBIN FPGAColumns']
    if CCDitem['GAIN Mode']=='High':
        CCDitem['DigGain'] = 0 
    elif CCDitem['GAIN Mode']=='Low':
        CCDitem['DigGain'] = 1   
    else:
        raise Exception('GAIN mode set to strange value')
       
    CCDitem['SigMode']=0
   # This should be read in, 0 should be high in output LM 200604  
#       CCDitem['']=CCDitem['']  
           
    CCDitem['read_from']='rac'
    try:
        CCDitem['reltime']=1.e-9*CCDitem['EXP Nanoseconds']
    except:
        try:
            CCDitem['reltime']=int(CCDitem['EXPTS'])+int(CCDitem['EXPTSS'])/2**16 
        except:
            raise Exception('No info on the relative time')



#        print(pathdir+str(CCD_image_data[i]['IMAGEFILE']) + '_data.npy')
    pngfile=rac_dir+str(CCDitem['Image File Name'])
    jsonfile=pngfile[0:-4]+'.json'
    try:
        CCDitem['IMAGE'] = np.float64(Image. open(pngfile))
        with open(jsonfile) as f:
            CCDitem['jsondata'] = json.load(f)
       
    except:    
        print('Warning, one image file seems corrupt and has been rmeoved')

    
        
        #Added temperature read in 
    
    if CCDitem['read_from']=='rac':    
        temperaturedata, relativetimedata=create_temperature_info_array(rac_dir+'HTR.csv')
    elif CCDitem['read_from']!='rac':
        temperaturedata=999
        relativetimedata=999
    
    #plot_full_temperature_info(temperaturedata,relativetimedata)
  
    CCDitem=add_temperature_info(CCDitem,temperaturedata,relativetimedata,labtemp)




    return CCDitem


def read_CCDitemsx(rac_dir,pathdir):
# reads data from all images (itemnumbers) in the rac file

    CCD_image_data=read_MATS_image(rac_dir+'images.json',pathdir)


    for CCDitem in CCD_image_data:
#        CCDitem=CCD_image_data[itemnumber]
     
    
    # The variables below are remain question marks
    #    CCDitem['Noverflow'] = Noverflow # FBINOV?
      #   CCDitem['Ending'] = Ending       #not found in rac 
        
        
    #    CCDitem['BC']=CCDitem['BC'][0] Bad culums ska vara en array
        CCDitem['CCDSEL']=CCDitem['CCDSEL'][0]
        CCDitem['EXPTS']=CCDitem['EXPTS'][0]
        CCDitem['EXPTSS']=CCDitem['EXPTSS'][0]
        CCDitem['FBINOV']=CCDitem['FBINOV'][0]
        CCDitem['FRAME']=CCDitem['FRAME'][0]
        CCDitem['JPEGQ']=CCDitem['JPEGQ'][0]
        CCDitem['LBLNK']=CCDitem['LBLNK'][0]
        CCDitem['NBC']=CCDitem['NBC'][0]
        CCDitem['NCOL']=CCDitem['NCOL'][0]
        CCDitem['NCSKIP']=CCDitem['NCSKIP'][0]
        CCDitem['NFLUSH']=CCDitem['NFLUSH'][0]
        CCDitem['NRBIN']=CCDitem['NRBIN'][0]        
        CCDitem['NROW']=CCDitem['NROW'][0]
        CCDitem['NRSKIP']=CCDitem['NRSKIP'][0]
        CCDitem['SID_mnemonic']=CCDitem['SID_mnemonic'][0]
        CCDitem['TBLNK']=CCDitem['TBLNK'][0]
        CCDitem['TEMP']=CCDitem['TEMP'][0]
        CCDitem['TEXPMS']=CCDitem['TEXPMS'][0]
        CCDitem['TIMING1']=CCDitem['TIMING1'][0] # Named Reserved1 in Georgis code /LM 20191115
        CCDitem['TIMING2']=CCDitem['TIMING2'][0] # Named Reserved2 in Georgis code /LM 20191115
        CCDitem['TIMING3']=CCDitem['TIMING3'][0] # Named VersionDate in Georgis code /LM 20191115
        CCDitem['VERSION']=CCDitem['VERSION'][0]
        CCDitem['WDWOV']=CCDitem['WDWOV'][0]
        CCDitem['ZERO']=CCDitem['ZERO'][0] 
    
        CCDitem['NColBinFPGA']=CCDitem['NColBinFPGA'][0]
        CCDitem['NColBinCCD']=CCDitem['NColBinCCD'][0]    
        CCDitem['DigGain']=CCDitem['DigGain'][0]
        CCDitem['TimingFlag']=CCDitem['TimingFlag'][0] 
        CCDitem['SigMode']=CCDitem['SigMode'][0]
        CCDitem['WinModeFlag']=CCDitem['WinModeFlag'][0]     
        CCDitem['WinMode']=CCDitem['WinMode'][0]  
        
    
    
        
        if int(CCDitem['CCDSEL'])==1:
            channel='IR1'
        elif int(CCDitem['CCDSEL'])==4:
            channel='IR2'
        elif int(CCDitem['CCDSEL'])==3:
            channel='IR3'
        elif int(CCDitem['CCDSEL'])==2:
            channel='IR4'
        elif int(CCDitem['CCDSEL'])==5:
            channel='UV1'
        elif int(CCDitem['CCDSEL'])==6:
            channel='UV2'
        elif int(CCDitem['CCDSEL'])==7:
            channel='NADIR'
        else:
            print('Error in CCDSEL, CCDSEL=',int(CCDitem['CCDSEL']))  
            
        
        CCDitem['channel']=channel
        
    

    return CCD_image_data

def read_CCDitems(rac_dir,labtemp=999):
    from math import log
    from get_temperature import create_temperature_info_array, add_temperature_info

# reads data from all images (itemnumbers) in the rac file

#    CCD_image_data=read_MATS_image(rac_dir+'images.json') #lest of dictionries
    CCD_image_data=read_MATS_image(rac_dir)


    for CCDitem in CCD_image_data:
#        CCDitem=CCD_image_data[itemnumber]
     
    
    # The variables below are remain question marks
    #    CCDitem['Noverflow'] = Noverflow # FBINOV?
      #   CCDitem['Ending'] = Ending       #not found in rac 
        

#        CCDitem['TIMING1']=CCDitem['TIMING1'][0] # Named Reserved1 in Georgis code /LM 20191115
#        CCDitem['TIMING2']=CCDitem['TIMING2'][0] # Named Reserved2 in Georgis code /LM 20191115
#        CCDitem['TIMING3']=CCDitem['TIMING3'][0] # Named VersionDate in Georgis code /LM 20191115


        # # The parameters below are missing in the new version of the rac files /LM 200603
        # CCDitem['SID_mnemonic']=CCDitem['SID_mnemonic'][0]
        # CCDitem['DigGain']=CCDitem['DigGain'][0]
        # CCDitem['TimingFlag']=CCDitem['TimingFlag'][0] 
        # CCDitem['SigMode']=CCDitem['SigMode'][0]
        # CCDitem['WinModeFlag']=CCDitem['WinModeFlag'][0]     
        # CCDitem['WinMode']=CCDitem['WinMode'][0]  
        
    
    
        
        if int(CCDitem['CCDSEL'])==1: #input CCDSEL=1
            channel='IR1'
        elif int(CCDitem['CCDSEL'])==4: #input CCDSEL=8
            channel='IR2'
        elif int(CCDitem['CCDSEL'])==3: #input CCDSEL=4
            channel='IR3'
        elif int(CCDitem['CCDSEL'])==2: #input CCDSEL=2
            channel='IR4'
        elif int(CCDitem['CCDSEL'])==5: #input CCDSEL=16
            channel='UV1'
        elif int(CCDitem['CCDSEL'])==6: #input CCDSEL=32
            channel='UV2'
        elif int(CCDitem['CCDSEL'])==7: #input CCDSEL=64
            channel='NADIR'
        else:
            print('Error in CCDSEL, CCDSEL=',int(CCDitem['CCDSEL']))  
            
        
        CCDitem['channel']=channel
#       Renaming of stuff. The names in the code here is based on the old rac extract file (prior to May 2020) rac_extract file works        
        CCDitem['id']=str(CCDitem['EXP Nanoseconds'])+'_'+str(CCDitem['CCDSEL'])
        
       # TODO LM June 2020: Change  all code so that the new names, i. CCDitem['NCBIN CCDColumns'] and CCDitem['NCBIN FPGAColumns'] are used instead of the old.
        try:
            CCDitem['NColBinCCD']
        except:
            CCDitem['NColBinCCD']=CCDitem['NCBIN CCDColumns']
        
        #CCDitem['NColBinFPGA']=CCDitem['NCBIN FPGAColumns']
        try:  
            CCDitem['NColBinFPGA']
        except: 
            CCDitem['NColBinFPGA']=log(CCDitem['NCBIN FPGAColumns'])/log(2)
            
        #del CCDitem['NCBIN FPGAColumns']
        if CCDitem['GAIN Mode']=='High':
            CCDitem['DigGain'] = 0 
        elif CCDitem['GAIN Mode']=='Low':
            CCDitem['DigGain'] = 1   
        else:
            raise Exception('GAIN mode set to strange value')
            
        CCDitem['SigMode']=0
        # This should be read in, 0 should be high in output LM 200604  
 #       CCDitem['']=CCDitem['']
        CCDitem['read_from']='rac'
        try:
            CCDitem['reltime']=1.e-9*CCDitem['EXP Nanoseconds']
        except:
            try:
                CCDitem['reltime']=int(CCDitem['EXPTS'])+int(CCDitem['EXPTSS'])/2**16 
            except:
                raise Exception('No info on the relative time')


        
        #Added temperature read in 
    
    if CCDitem['read_from']=='rac':    
        temperaturedata, relativetimedata=create_temperature_info_array(rac_dir+'HTR.csv')
    elif CCDitem['read_from']!='rac':
        temperaturedata=999
        relativetimedata=999
    
    #plot_full_temperature_info(temperaturedata,relativetimedata)

    for CCDitem in CCD_image_data:

        CCDitem=add_temperature_info(CCDitem,temperaturedata,relativetimedata,labtemp)
#        timestamp=epoch+datetime.timedelta(0,CCDitem['reltime'])



                
    return CCD_image_data

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value



def readimg(filename):

    data_arr=np.fromfile(filename, dtype='uint16')#check endianess, uint16 instead of ubit16 seems to work
 
    #convert header to binary
    #header_bin = [bin(np.double(data_arr[i])) for i in range(0,12)]#check if at least 16 bits, check if indexing correct
    header_bin = np.asarray([bin(data_arr[i]) for i in range(0,12)])#this is a string 
    #change format of header_bin elements to be formatted like matlab char array
    for i in range(0,len(header_bin)):
        header_bin[i]=header_bin[i][2:].zfill(16)
    #read header
    Frame_count = int(header_bin[0],2)
    NRow = int(header_bin[1],2)
    NRowBinCCD = int(header_bin[2][10:16],2)
    NRowSkip = int(header_bin[3][8:16],2)
    NCol = int(header_bin[4],2)
    NColBinFPGA = int(header_bin[5][2:8],2)
    NColBinCCD = int(header_bin[5][8:16],2)
    NColSkip = int(header_bin[6][5:16],2)
    N_flush = int(header_bin[7],2)
    Texposure_MSB = int(header_bin[8],2)
    Texposure_LSB = int(header_bin[9],2)
    Gain = int(header_bin[10],2)
    SignalMode = Gain >> 12 & 1   
    Temperature_read = int(header_bin[11],2)


    #read image
    if len(data_arr)< NRow*(NCol+1)/(2**(NColBinFPGA)):#check for differences in python 2 and 3
        img_flag = 0
        image = 0
        Noverflow = 0
        BlankLeadingValue = 0
        BlankTrailingValue = 0
        ZeroLevel = 0
        
        Reserved1 = 0
        Reserved2 = 0
        
        Version = 0
        VersionDate = 0
        NBadCol = 0
        BadCol = 0
        Ending = 'Wrong size'
    else:
        img_flag = 1
        image = np.reshape(np.double(data_arr[11+1:NRow*(NCol+1)+12]), (NCol+1,NRow))
        image = np.matrix(image).getH()
    
        #Trailer
        trailer_bin = np.asarray([bin(data_arr[i]) for i in range(NRow*(NCol+1)+12,len(data_arr))])
        Noverflow = int(trailer_bin[0],2)
        BlankLeadingValue = int(trailer_bin[1],2)
        BlankTrailingValue = int(trailer_bin[2],2)
        ZeroLevel = int(trailer_bin[3],2)
        
        Reserved1 = int(trailer_bin[4],2)
        Reserved2 = int(trailer_bin[5],2)
        
        Version = int(trailer_bin[6],2)
        VersionDate = int(trailer_bin[7],2)
        NBadCol = int(trailer_bin[8],2)
        BadCol = []
        Ending = int(trailer_bin[-1],2)
        
        if NBadCol > 0:
            BadCol = np.zeros(NBadCol)
            for k_bc in range(0,NBadCol):
                BadCol[k_bc] = int(trailer_bin[9+k_bc],2)#check if this actually works
    
    #original matlab code uses structured array, as of 20-03-2019 implementation as dictionary seems to be more useful choice
    #decision might depend on further (computational) use of data, which is so far unknown to me
    header = {}
    header['Size'] = len(data_arr)
    header['FRAME'] = Frame_count
    header['NROW'] = NRow
    header['NRBIN'] = NRowBinCCD
    header['NRSKIP'] = NRowSkip
    header['NCOL'] = NCol
    header['NColBinFPGA'] = NColBinFPGA
    header['NColBinCCD'] = NColBinCCD
    header['NCSKIP'] = NColSkip
    header['NFLUSH'] = N_flush
    header['TEXPMS'] = Texposure_LSB + Texposure_MSB*2**16
    header['DigGain'] = Gain & 0b1111
    header['SigMode'] = SignalMode
    header['TEMP'] = Temperature_read
    header['Noverflow'] = Noverflow # FBINOV?
    header['LBLNK'] = BlankLeadingValue
    header['TBLNK'] = BlankTrailingValue
    header['ZERO'] = ZeroLevel
    header['TIMING1'] = Reserved1
    header['TIMING2'] = Reserved2
    header['VERSION'] = Version
    header['TIMING3'] = VersionDate
    header['NC'] = NBadCol
    header['BC'] = BadCol
    header['Ending'] = Ending            
    header['CCDSEL'] = 1 #LM CCDSEL 1 is really IR1, so this is incorrent but the KTH test CCD has no value
    header['channel'] = 'KTH test channel' 
    return image, header, img_flag


def readimage_create_CCDitem(path, file_number): #reads file from georigis stuff LM20191113
    
    filename = '%sF_0%02d/D_0%04d' % (path, np.floor(file_number/100),file_number)
    data_arr=np.fromfile(filename, dtype='uint16')#check endianess, uint16 instead of ubit16 seems to work
 
    #convert header to binary
    #header_bin = [bin(np.double(data_arr[i])) for i in range(0,12)]#check if at least 16 bits, check if indexing correct
    header_bin = np.asarray([bin(data_arr[i]) for i in range(0,12)])#this is a string 
    #change format of header_bin elements to be formatted like matlab char array
    for i in range(0,len(header_bin)):
        header_bin[i]=header_bin[i][2:].zfill(16)
    #read header
    Frame_count = int(header_bin[0],2)
    NRow = int(header_bin[1],2)
    NRowBinCCD = int(header_bin[2][10:16],2)
    NRowSkip = int(header_bin[3][8:16],2)
    NCol = int(header_bin[4],2)
    NColBinFPGA = int(header_bin[5][2:8],2)
    NColBinCCD = int(header_bin[5][8:16],2)
    NColSkip = int(header_bin[6][5:16],2)
    N_flush = int(header_bin[7],2)
    Texposure_MSB = int(header_bin[8],2)
    Texposure_LSB = int(header_bin[9],2)
    Gain = int(header_bin[10],2)
    SignalMode = Gain >> 12 & 1      
    Temperature_read = int(header_bin[11],2)


    #read image
    if len(data_arr)< NRow*(NCol+1)/(2**(NColBinFPGA)):#check for differences in python 2 and 3
        img_flag = 0
        image = 0
        Noverflow = 0
        BlankLeadingValue = 0
        BlankTrailingValue = 0
        ZeroLevel = 0
        
        Reserved1 = 0
        Reserved2 = 0
        
        Version = 0
        VersionDate = 0
        NBadCol = 0
        BadCol = 0
        Ending = 'Wrong size'
    else:
        img_flag = 1
        image = np.reshape(np.double(data_arr[11+1:NRow*(NCol+1)+12]), (NCol+1,NRow))
        image = np.matrix(image).getH()
    
        #Trailer
        trailer_bin = np.asarray([bin(data_arr[i]) for i in range(NRow*(NCol+1)+12,len(data_arr))])
        Noverflow = int(trailer_bin[0],2)
        BlankLeadingValue = int(trailer_bin[1],2)
        BlankTrailingValue = int(trailer_bin[2],2)
        ZeroLevel = int(trailer_bin[3],2)
        
        Reserved1 = int(trailer_bin[4],2)
        Reserved2 = int(trailer_bin[5],2)
        
        Version = int(trailer_bin[6],2)
        VersionDate = int(trailer_bin[7],2)
        NBadCol = int(trailer_bin[8],2)
        BadCol = []
        Ending = int(trailer_bin[-1],2)
        
        if NBadCol > 0:
            BadCol = np.zeros(NBadCol)
            for k_bc in range(0,NBadCol):
                BadCol[k_bc] = int(trailer_bin[9+k_bc],2)#check if this actually works
    
    #original matlab code uses structured array, as of 20-03-2019 implementation as dictionary seems to be more useful choice
    #decision might depend on further (computational) use of data, which is so far unknown to me. /Frank
    CCDitem = {}
#    CCDitem['Size'] = len(data_arr)
    CCDitem['FRAME'] = Frame_count
    CCDitem['NROW'] = NRow
    CCDitem['NRBIN'] = NRowBinCCD
    CCDitem['NRSKIP'] = NRowSkip
    CCDitem['NCOL'] = NCol
    CCDitem['NColBinFPGA'] = NColBinFPGA 
    CCDitem['NColBinCCD'] = NColBinCCD
    CCDitem['NCSKIP'] = NColSkip
    CCDitem['NFLUSH'] = N_flush
    CCDitem['TEXPMS'] = Texposure_LSB + Texposure_MSB*2**16
    CCDitem['DigGain'] = Gain & 0b1111
    CCDitem['SigMode'] = SignalMode 
    CCDitem['TEMP'] = Temperature_read
    CCDitem['Noverflow'] = Noverflow # FBINOV?
    CCDitem['LBLNK'] = BlankLeadingValue  
    CCDitem['TBLNK'] = BlankTrailingValue
    CCDitem['ZERO'] = ZeroLevel
    CCDitem['TIMING1'] = Reserved1
    CCDitem['TIMING2'] = Reserved2 
    CCDitem['VERSION'] = Version
    CCDitem['TIMING3'] = VersionDate 
    CCDitem['NBC'] = NBadCol
    CCDitem['BC'] = BadCol
    CCDitem['Ending'] = Ending       #not found in rac 
    
    CCDitem['IMAGE']=image
    
    CCDitem['CCDSEL'] = 1 # Note that this is incorrect but the KTH test CCD is unknown    
    CCDitem['channel'] = 'KTH test channel' 
    
    
    return CCDitem, img_flag



def readracimg(filename):
#   Linda Megner; function to read in from rac file but yield similar result 
#    as when read in by readimg . Note that this header has more info.
        
    image, metadata = imagereader.read_MATS_image(filename) 
    image=np.float64(image)

    header = {}
#    header['Size'] = image.size
#   LM: Note this is NOT equivalent to what is in readim    
#    header['Size'] = len(data_arr). This variable not needed

  
    header['Frame_count'] = metadata['FRAME'][0]
    header['NRow'] = metadata['NROW'][0]
    header['NRowBinCCD'] = metadata['NRBIN'][0]
    header['NRowSkip'] = metadata['NRSKIP'][0]
    header['NCol'] = metadata['NCOL'][0]
    header['NColBinFPGA'] = metadata['NCBIN'][0] >> 8 & 0b1111
    header['NColBinCCD'] = metadata['NCBIN'][0] & 0b11111111
    header['NColSkip'] = metadata['NCSKIP'][0]
    header['N_flush'] = metadata['NFLUSH'][0]
    header['Texposure'] = metadata['TEXPMS'][0]
    header['DigGain'] = metadata['GAIN'] & 0b1111
    header['SignalMode'] = metadata['GAIN'] >> 12 & 1          
     
    
    header['Temperature_read'] = metadata['TEMP'][0] #LM what is this the temperature of and how do I convert it? 
#  Temperature is measured in the CRBD – readout ADC, so closer to OBC.
#It’s a 15 bit value from 2.048V (+/-0.5%) reference.
#So 4307/32768*2048 = 269 mV
#According to datasheet ADC outputs 273mV @ 25°C with 0.85 mV/°C sensitivity. So it’s approximately 20-21°C when the image was taken.
    header['Noverflow'] =  metadata['FBINOV'][0]  
    header['BlankLeadingValue'] =  metadata['LBLNK'][0]  
    header['BlankTrailingValue'] =  metadata['TBLNK'][0]  
    header['ZeroLevel'] =  metadata['ZERO'][0] 

    
    header['Reserved1'] = metadata['TIMING1'][0] #Check that this is correct by comparing old and new software ICD
    header['Reserved2'] = metadata['TIMING2'][0] #Check that this is correct by comparing old and new software ICD
    header['Version'] = metadata['VERSION'][0]
    header['VersionDate'] = metadata['TIMING3'][0] #Check that this is correct by comparing old and new software ICD
    header['NBadCol'] = metadata['NBC'][0]
    header['BadCol'] = metadata['BC']
#    header['Ending'] = 64175    NO INFO IN RAC FILE metadata 
    
#   More information not used in readimg
    header['CCDSEL'] = metadata['CCDSEL'][0]  
    header['EXPTS'] = metadata['EXPTS'][0]   
    header['EXPTSS'] = metadata['EXPTSS'][0]   
    header['JPEGQ'] = metadata['JPEGQ'][0]  
    header['SID_mnemonic'] = metadata['SID_mnemonic'][0] 
    header['WDW'] = metadata['WDW'][0]   
    header['WDWOV'] = metadata['WDWOV'][0]
    
    header['TimingFlag'] =  metadata['GAIN'] >> 8 & 1
      
#    img_flag=1 #LM is this needed? Ask Georgi
    return image, header





def readimgpath(path, file_number, plot):

    filename = '%sF_0%02d/D_0%04d' % (path, np.floor(file_number/100),file_number)
    image, header, img_flag = readimg(filename)

    if plot>0:
        mean_img=np.mean(image)    
        plt.imshow(image, cmap='viridis', vmin=mean_img-100, vmax=mean_img+100)
        plt.title('CCD image')
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        plt.show()#equivalent to hold off
        
        print(header)
        time.sleep(0.1)
    
    return image, header


def read_txtfile(filepath):    
    dict = {}
    with open(filepath, "r") as f:
        for line in f:
            (key, val) = line.split("=")
            if key=='id':
                dict['ID'] = val[1:-2] #For some reason a space has been added before and an ewnd og line character in the end - remove these
            else:
                dict[key] = int(val)
    return dict
    
def read_txtfile_create_CCDitem(filepath):
    # function used to read txt output from image viewer
    CCDitem=read_txtfile(filepath)

# Extract variables from certain bits within the same element, see 6.4.1 Software ICD /LM 20191115               

    CCDitem['NColBinFPGA'] =  CCDitem['NCBIN'] >> 8 & 0b1111
    CCDitem['NColBinCCD'] = CCDitem['NCBIN'] & 0b11111111
    del CCDitem['NCBIN']
    CCDitem['DigGain'] = CCDitem['GAIN'] & 0b1111
    CCDitem['TimingFlag'] = CCDitem['GAIN'] >> 8 & 1
    CCDitem['SigMode'] =CCDitem['GAIN'] >> 12 & 1          
    del CCDitem['GAIN']
    CCDitem['WinModeFlag']=CCDitem['WDW'] >> 7 & 1
    CCDitem['WinMode']=CCDitem['WDW'] & 0b111
    del CCDitem['WDW']
    # Hack to make image viewing output the same as the rac file outputs
    CCDitem['id']=CCDitem['ID']
    del CCDitem['ID']
    
    
    print('The following values may be incorrect:  NColBinFPGA, NColBinCCD DigGain TimingFlag SigMode inModeFlag WinMode')
    if int(CCDitem['CCDSEL'])==1:
        channel='IR1'
    elif int(CCDitem['CCDSEL'])==4:
        channel='IR2'
    elif int(CCDitem['CCDSEL'])==3:
        channel='IR3'
    elif int(CCDitem['CCDSEL'])==2:
        channel='IR4'
    elif int(CCDitem['CCDSEL'])==5:
        channel='UV1'
    elif int(CCDitem['CCDSEL'])==6:
        channel='UV2'
    elif int(CCDitem['CCDSEL'])==7:
        channel='NADIR'
    else:
        print('Error in CCDSEL, CCDSEL=',int(CCDitem['CCDSEL']))  
    CCDitem['channel']=channel
    
    return CCDitem