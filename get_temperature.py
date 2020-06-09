#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:18:00 2020

@author: lindamegner
"""

import json
import datetime
import numpy as np

import scipy.interpolate

def read_MATS_packets(filename):
    json_file = open(filename,'r')
    packet_data = json.load(json_file)
    json_file.close
    return packet_data

def get_time(relativetime,endtime):
    epoch=datetime.datetime(1984,10,1)-datetime.timedelta(0,endtime)
    timestamp=epoch+datetime.timedelta(0,relativetime)
    return timestamp

def add_temperature_info(CCDitem,temperaturedata, relativetimedata, temperature=-999):
    #Find the temperature of the CCDs. If not read from rac set the temperature.
    if CCDitem['read_from']=='rac':        
        #find the closest time when heater settings have been recorded. Could be changed to interpolate.
        ind = (np.abs(relativetimedata-CCDitem['reltime'])).argmin()  
        HTR1A=temperaturedata[ind,0]
        HTR1B=temperaturedata[ind,1]
        HTR2A=temperaturedata[ind,2]
        HTR2B=temperaturedata[ind,3]
        HTR8A=temperaturedata[ind,4]
        HTR8B=temperaturedata[ind,5]
        temperature=HTR8B
 #       temperature=[HTR1A,HTR1B,HTR2A,HTR2B,HTR8A,HTR8B]    #Temperatere taken from UV channel at the moment
    elif CCDitem['read_from']=='imgview' or CCDitem['read_from']=='KTH': #Take temperature from ADC
        #Check ADC temperature. This will not be part of the calibration routine but is used as a sanity test.  
        #273mV @ 25°C with 0.85 mV/°C
        ADC_temp_in_mV=int(CCDitem['TEMP'])/32768*2048 
        ADC_temp_in_degreeC=1./0.85*ADC_temp_in_mV-296
        temperature=ADC_temp_in_degreeC #Change this to read temperature sensors from rac file
        #temperature=-18 #-18C is measured at TVAC tests in August 2019    
    else:
        raise Exception('read_from needs to be rac, imgview or KTH')
#  #      print('Warning: No temperature infromation. Temperature is set in code which will affect dark current reduction')
    CCDitem['temperature']=temperature
    return CCDitem

def create_temperature_info_array(filename):
    import pandas as pd

    df = pd.read_csv(filename, skiprows=[0]) 
    package_data=df.to_dict('records')
    
    
    lastpackage=package_data[len(package_data)-1]
    #endtime=int(lastpackage['DFH_CUC_time_seconds'])+int(lastpackage['DFH_CUC_time_fraction'])/2**16
    endtime=float(1.e-9*lastpackage['TMHeaderNanoseconds'])
    
    # Find all packages with heater information
    packdicts= [d for d in package_data if d['SID']=='HTR']
    
    

    templist=[]
    index=-1
    HTRdata=np.empty((len(packdicts),6))
    relativetimedata=np.empty((len(packdicts)))
    # Build 2D array with heater information. 6 entires (heater channels) for every package number.  
    for packdict in packdicts:
        index=index+1
 #       relativetime=int(packdict['DFH_CUC_time_seconds'])+int(packdict['DFH_CUC_time_fraction'])/2**16
        relativetime=float(1.e-9*packdict['TMHeaderNanoseconds'])
        timestamp=get_time(relativetime,endtime)
#        print(timestamp)
        
    #    packdict['timestamp']=timestamp    
    

        
    #    templist.append(temp_in_counts[0])
        HTRdata[index,0]=packdict['HTR1A']
        HTRdata[index,1]=packdict['HTR1B']
        HTRdata[index,2]=packdict['HTR2A']
        HTRdata[index,3]=packdict['HTR2B']
        HTRdata[index,4]=packdict['HTR8A']
        HTRdata[index,5]=packdict['HTR8B']
        
        relativetimedata[index]=relativetime
        
    

        
        
    #temp_readout=np.array(templist)
    
    # #Heater Output drive
    # xaxis=np.array([0, 1023, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 4095])
    # htr1_dc =np.array([0,0, 7.1, 12.9, 18.4, 23.7, 28.8, 33.9, 38.8, 43.7, 48.5, 53.3, 58.2, 63.1, 68.1, 73.2, 78.5, 83.9, 89.6, 95.6, 100, 100])
    # htr2_dc =np.array([0, 0, 7.2, 13, 18.5, 23.8, 28.9, 34, 38.9, 43.7, 48.6, 53.4, 58.2, 63.1, 68.1, 73.2, 78.5, 83.9, 89.6, 95.6, 100, 100])
    # htr7_dc =np.array([0, 0, 7.1, 12.9, 18.4, 23.7, 28.9, 33.9, 38.8, 43.6, 48.5, 53.3, 58.2, 63, 68, 73.1, 78.4, 83.8, 89.5, 95.5, 100, 100])
    # htr8_dc =np.array([0, 0, 7.1, 12.9, 18.4, 23.7, 28.8, 33.8, 38.7, 43.6, 48.4, 53.2, 58.1, 63, 68, 73.1, 78.3, 83.8, 89.4, 95.5, 100, 100])
    
    # htr_dc = [htr1_dc; htr2_dc; htr7_dc; htr8_dc];
    
    # figure();
    # plot(xaxis, htr_dc, '.-');
    # grid('on');
    # legend('HTR1', 'HTR2', 'HTR7', 'HTR8');
    # xlabel('Output Drive');
    # ylabel('Duty cycle (%)');
    
    # %% Heater temperature sensors
    # D = 0:100:2^12-1;
    # Vadc = 2.5 / (2^12 - 1) * D;
    # Rntc = 3.3 * 3900 ./ Vadc - 3900;
    

    # # From datasheet, now in Level0 calibration
    # temp = np.arange(-55,85,5)
    # RtR25 =np.array([96.3, 67.01, 47.17, 33.65, 24.26, 17.7, 13.04, 9.707, 7.293, 5.533, 4.232, 3.265,
    # 		2.539, 1.99, 1.571, 1.249, 1.0000, 0.8057, 0.6531, 0.5327, 0.4369, 0.3603, 0.2986,
    # 		0.2488, 0.2083, 0.1752, 0.1481, 0.1258])
    
    
    # Rntc = 10000 * RtR25
    # Vadc = (3.3 * 3900) / (Rntc + 3900)
    # D = (2**12 - 1) * Vadc / 2.5
    
    
    # HTRtemp=np.empty(HTRdata.shape)
    # y_interp = scipy.interpolate.interp1d(D, temp)
    # for i in range(0,6): 
    #     HTRtemp[:,i]=y_interp(HTRdata[:,i])
    


    
    return HTRdata, relativetimedata