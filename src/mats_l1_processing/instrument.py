# -*- coding: utf-8 -*-
"""Classes containing information about the MATS instrument.

This module has a CCD class which contains the information about a specific CCD 
in MATS such as dark-current patterns, non-linearity response and flat fields. Another
class is used as a container for several CCDs. Finally a class for a generalized non-linearity
is included. 

"""

import os
import toml
import numpy as np
import pickle
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
from scipy.ndimage import median_filter
import sqlite3


class CCD:
    """Class to represent a single physical CCD on MATS, a.k.a CCDunit

    The class contains attributes and methods for extracting calibration parameters. Most 
    parameters are dependent on the signalmode used in the CCD which can either be high signal 
    mode(HSM) or low signal mode(LSM).

    Attributes:
        channel (str): CCD channel
        CCDID (int): CCDID
        channelnumbe (int): Numeric representation of channel
        CPRU_Port (str): CPRU Port  of channel
        KTH_name (str): KTH name of channel
        OHB_naming (str): OHB name of channel
        OHB_marker_tag (int): OHB marker tag of channel
                
        dc_zero_avr_HSM:
        dc_zero_std_HSM:
        dc_zero_avr_LSM:
        dc_zero_std_LSM:

        image_HSM:
        image_LSM:

        ro_avr_HSM:
        ro_std_HSM:
        alpha_avr_HSM: average electrons per count in HSM
        alpha_std_HSM: std of electrons per count in HSM

        ro_avr_LSM:
        ro_std_LSM: 
        alpha_avr_LSM: average electrons per count in LSM
        alpha_std_LSM: std of electrons per count in LSM

        log_a_avr_HSM: dark current parameter HSM
        log_a_std_HSM: dark current parameter HSM
        log_b_avr_HSM: dark current parameter HSM
        log_b_std_HSM: dark current parameter HSM

        log_a_avr_LSM: dark current parameter LSM
        log_a_std_LSM: dark current parameter LSM
        log_b_avr_LSM: dark current parameter LSM
        log_b_std_LSM: dark current parameter LSM

        log_a_img_avr_LSM: 2D dark current parameter LSM
        log_a_img_err_LSM: 2D dark current parameter LSM
        log_b_img_avr_LSM: 2D dark current parameter LSM
        log_b_img_err_LSM: 2D dark current parameter LSM

        log_a_img_avr_HSM: 2D dark current parameter HSM
        log_a_img_err_HSM: 2D dark current parameter HSM
        log_b_img_avr_HSM: 2D dark current parameter HSM
        log_b_img_err_HSM: 2D dark current parameter HSM

        flatfield_HSM (np.array): flatfield image for HSM 
        flatfield_LSM (np.array): flatfield image for LSM 

        non_linearity_pixel (nonLinearity): Linearity of an average pixel
        non_linearity_sumrow (nonLinearity): Linearity of the shift register
        non_linearity_sumwell (nonLinearity): Linerarity of the summation well

        ampcorrection (float): correction for pre-amplification

    """

    def __init__(self, channel, calibration_file, start_datetime: datetime = datetime(2000,1,1), end_datetime: datetime = datetime(3000,1,1)):

        """Init method for CCD class

        Args:
            channel (str): Name of channel
            calibration_file (str): calibration file containing paths to calibration files
            start_datetime (datetime): datetime on which start extraction calibration parameters (default: None) 
            end_datetime (datetime): datetime on which to end extraction calibration parameters (default: None) 

        """

        def get_sqlite_data(
            filename: str,
            table: str,
        ) -> pd.DataFrame:
            if filename.startswith("s3://"):
                from boto3 import client
                from tempfile import gettempdir
                s3 = client("s3")
                bucket = filename[5:].split("/")[0]
                key = filename[6 + len(bucket):]
                db_name = key.split("/")[-1]
                db_path = f"{gettempdir()}/{db_name}"
                if not os.path.exists(db_path):
                    s3.download_file(bucket, key, db_path)
                return get_sqlite_data(db_path, table)
            conn = sqlite3.connect(filename)
            query = f'''
                SELECT * FROM {table}
                WHERE datetime BETWEEN '{start_datetime}' AND '{end_datetime}'
                AND channel = '{self.channel}'
            '''
            data = pd.read_sql_query(query, conn)
            data['datetime'] = pd.to_datetime(data['datetime'])
            conn.close()
            return data

        self.channel = channel
        if channel == "IR1":
            CCDID = 16
            channelnumber = 1
            CPRU_Port = "A0"
            KTH_name = "FM2"
            OHB_naming = "CCD_2"
            OHB_marker_tag = 14
        elif channel == "IR2":
            CCDID = 17
            channelnumber = 4
            CPRU_Port = "A3"
            KTH_name = "FM3"
            OHB_naming = "CCD_3"
            OHB_marker_tag = 15
        elif channel == "IR3":
            CCDID = 18
            channelnumber = 3
            CPRU_Port = "A2"
            KTH_name = "FM4"
            OHB_naming = ""
            OHB_marker_tag = ""
        elif channel == "IR4":
            CCDID = 19
            channelnumber = 2
            CPRU_Port = "A1"
            KTH_name = "FM8"
            OHB_naming = ""
            OHB_marker_tag = ""
        elif channel == "NADIR":
            CCDID = 20
            channelnumber = 7
            CPRU_Port = "B2"
            KTH_name = "FM7"
            OHB_naming = 50
            OHB_marker_tag = "NADIR_CAM"
        elif channel == "UV1":
            CCDID = 21
            channelnumber = 5
            CPRU_Port = "B0"
            KTH_name = "FM5"
            OHB_naming = "CCD_5"
            OHB_marker_tag = 32
        elif channel == "UV2":
            CCDID = 22
            channelnumber = 6
            CPRU_Port = "B1"
            KTH_name = "FM6"
            OHB_naming = "CCD_4"
            OHB_marker_tag = 49
        elif channel == "KTH test channel":
            CCDID = 16
            channelnumber = None
            CPRU_Port = None
            KTH_name = None
            OHB_naming = None
            OHB_marker_tag = None

        self.CCDID = CCDID
        self.channelnumber = channelnumber
        self.CPRU_Port = CPRU_Port
        self.KTH_name = KTH_name
        self.OHB_naming = OHB_naming
        self.OHB_marker_tag = OHB_marker_tag

        calibration_data = toml.load(calibration_file)
        
        self.default_temp = calibration_data["darkcurrent"]["default_temp"]
        # Limit to choose 1D or 2D dark current subtraction
        self.dc_2D_limit=calibration_data["darkcurrent"]["dc_2D_limit"]

        # Read in Gabriels calibration data from a .mat file
        filename = (
            calibration_data["darkcurrent"]["folder"]
            + "FM0"
            + str(CCDID)
            + "_CCD_DC_calibration.mat"
        )

        mat = loadmat(filename)

        self.dc_zero_avr_HSM = mat["dc_zero_avr_HSM"]
        self.dc_zero_std_HSM = mat["dc_zero_std_HSM"]
        self.dc_zero_avr_LSM = mat["dc_zero_avr_LSM"]
        self.dc_zero_std_LSM = mat["dc_zero_std_LSM"]

        self.image_HSM = mat["image_HSM"]
        self.image_LSM = mat["image_LSM"]

        self.ro_avr_HSM = mat["ro_avr_HSM"]
        self.ro_std_HSM = mat["ro_std_HSM"]
        self.alpha_avr_HSM = mat["alpha_avr_HSM"]
        self.alpha_std_HSM = mat["alpha_std_HSM"]

        self.ro_avr_LSM = mat["ro_avr_LSM"]
        self.ro_std_LSM = mat["ro_std_LSM"]
        self.alpha_avr_LSM = mat["alpha_avr_LSM"]
        self.alpha_std_LSM = mat["alpha_std_LSM"]
        
        # 0D dark current subtraction stuff
        self.log_a_avr_HSM = mat["log_a_avr_HSM"]
        self.log_a_std_HSM = mat["log_a_std_HSM"]
        self.log_b_avr_HSM = mat["log_b_avr_HSM"]
        self.log_b_std_HSM = mat["log_b_std_HSM"]

        self.log_a_avr_LSM = mat["log_a_avr_LSM"]
        self.log_a_std_LSM = mat["log_a_std_LSM"]
        self.log_b_avr_LSM = mat["log_b_avr_LSM"]
        self.log_b_std_LSM = mat["log_b_std_LSM"]

        # 2D dark current subtraction stuff
        self.log_a_img_avr_LSM = median_filter(mat["log_a_img_avr_LSM"], size=3)
        self.log_a_img_err_LSM = median_filter(mat["log_a_img_err_LSM"], size=3)
        self.log_b_img_avr_LSM = median_filter(mat["log_b_img_avr_LSM"], size=3)
        self.log_b_img_err_LSM = median_filter(mat["log_b_img_err_LSM"], size=3)

        self.log_a_img_avr_HSM = median_filter(mat["log_a_img_avr_HSM"], size=3)
        self.log_a_img_err_HSM = median_filter(mat["log_a_img_err_HSM"], size=3)
        self.log_b_img_avr_HSM = median_filter(mat["log_b_img_avr_HSM"], size=3)
        self.log_b_img_err_HSM = median_filter(mat["log_b_img_err_HSM"], size=3)


        # Flatfields
        if channel=="NADIR":
            self.flatfield_HSM =np.ones(self.log_a_img_avr_HSM.shape)
        else:
            self.flatfield_HSM = np.load(
                calibration_data["flatfield"]["flatfieldfolder"]
                + "flatfield_"
                + channel
                + "_HSM.npy"
            )

        # Non-linearity
        if channel!="NADIR":
            with open(calibration_data["linearity"]["pixel"]
                + "_" + str(self.channelnumber)
                + ".pkl", 'rb') as fp:
    
                self.non_linearity_pixel = pickle.load(fp)
    
            with open(calibration_data["linearity"]["sumrow"]
                + "_" + str(self.channelnumber)
                + ".pkl", 'rb') as fp:
    
                self.non_linearity_sumrow = pickle.load(fp)
    
            with open(calibration_data["linearity"]["sumwell"]
                + "_" + str(self.channelnumber)
                + ".pkl", 'rb') as fp:
    
                self.non_linearity_sumwell = pickle.load(fp)
            
            if 'tables' in calibration_data["linearity"]:
                self.tablefolder = calibration_data["linearity"]["tables"]
                self.tables = pd.read_csv(self.tablefolder + 'tables.csv')
            else:
                self.tables = None

        # Amplification correction for UV channels
        if self.channel == "UV1" or self.channel == "UV2":
            self.ampcorrection = (
                3 / 2
            )
        else:
            self.ampcorrection = 1
        
        # Absolute and relative calibration constants
        
        df = pd.read_csv(calibration_data["abs_rel_calib"]["abs_rel_calib_constants"], comment="#",
                         skipinitialspace=True, skiprows=()) 
        
        if self.channel=='IR1':
            self.cal_fact_HSM=df["abs_ir1"][0]
            self.cal_fact_LSM=df["abs_ir1"][1]
        elif self.channel=='IR2':
            self.cal_fact_HSM=df["abs_ir2"][0]
            self.cal_fact_LSM=df["abs_ir2"][1]
        elif self.channel=='IR3':
            self.cal_fact_HSM=df["abs_ir3"][0]
            self.cal_fact_LSM=df["abs_ir3"][1]
        elif self.channel=='IR4':
            self.cal_fact_HSM=df["abs_ir4"][0]
            self.cal_fact_LSM=df["abs_ir4"][1]
        elif self.channel=='UV1':
            self.cal_fact_HSM=df["abs_uv1"][0]
            self.cal_fact_LSM=df["abs_uv1"][1]
        elif self.channel=='UV2':
            self.cal_fact_HSM=df["abs_uv2"][0]
            self.cal_fact_LSM=df["abs_uv2"][1]
        elif self.channel=='NADIR':
            self.cal_fact_HSM=df["abs_nadir"][0]
            self.cal_fact_LSM=df["abs_nadir"][1]

        # Read in without pandas
        # with open(calibration_data["calibration"]["calibration_constants"]) as f:
        #     h = [float(x) for x in next(f).split()] # read first line
        #     if self.channel=='IR1':
        #         self.cal_fact=h[0]
        #     elif self.channel=='IR2':
        #         self.cal_fact=h[1] 
        #     elif self.channel=='IR3':
        #         self.cal_fact=h[2]
        #     elif self.channel=='IR4':
        #         self.cal_fact=h[3]
        #     elif self.channel=='UV1':
        #         self.cal_fact=h[4]
        #     elif self.channel=='UV2':
        #         self.cal_fact=h[5]
        #     elif self.channel=='NADIR':
        #         self.cal_fact=h[6]
    
        #quaternion
        filename=calibration_data['pointing']['qprime']+'qprime.csv'
        qprimes=np.loadtxt(filename,delimiter=',',dtype={'names':('Channel','q0','q1','q2','q3'),
                            'formats':('S5','f4','f4','f4','f4')})
        self.qprime = np.array([(d['q0'],d['q1'],d['q2'],d['q3']) for d in qprimes if d['Channel'].decode('utf8')==self.channel][0])

        # artifact correction
        if channel=="NADIR":
            filename = calibration_data['artifact']['nadir']
            self.artifact_masks = pd.read_pickle(filename)
            
        else :        
            filename = calibration_data['artifact']['blank']
            self.artifact_masks = pd.read_pickle(filename)

        # single event correction
        filename = calibration_data['hot_pixels']['single_events']
        self.single_event = get_sqlite_data(filename, "SingleEvents")

        # hot pixel correction
        filename = calibration_data['hot_pixels']['hot_pixels']
        self.hot_pixels = get_sqlite_data(filename, "hotpixelmaps")
        # Convert the HPM to a NumPy array
        self.hot_pixels['HPM'] = self.hot_pixels['HPM'].apply(pickle.loads)
                
    def calib_denominator(self, mode): 
        """Get calibration constant that should be divided by to get unit 10^15 ph m-2 s-1 str-1 nm-1.

        Args:
            mode (str): Gain mode/ Signal mode for CCD 

        Returns:
            calib_denominator: float

        """

        if mode == "High":
            calib_denominator=self.cal_fact_HSM
        elif mode == "Low":
            calib_denominator=self.cal_fact_LSM
        else:
            raise ValueError('Mode must be "High" or "Low"')
        return calib_denominator
        
        
    def getrawdark(self, log_a, log_b, T):
        #calculates dark current in electrons/s
        rawdark = 10 ** (log_a * T + log_b)
        return rawdark
    

    def darkcurrent(self, T, mode):  # electrons/s
        """Get an average dark current for a channel. 

        This method can be used of the dark current signal is very low, since using the 2D functions become noisy then.

        Args:
            T (float): Temperature of CCD
            mode (str): Gain mode/ Signal mode for CCD 

        Returns:
            darkcurrent(float): average dark current of the CCD

        """

        if mode == 'High':
            log_a_avr=self.log_a_avr_HSM
            log_b_avr=self.log_b_avr_HSM
            log_a_std=self.log_a_std_HSM
            log_b_std=self.log_b_std_HSM
        elif mode == 'Low':
            log_a_avr=self.log_a_avr_LSM
            log_b_avr=self.log_b_avr_LSM 
            log_a_std=self.log_a_std_LSM
            log_b_std=self.log_b_std_LSM           
        else:
            raise Exception("Undefined mode")

        darkcurrent = self.getrawdark(log_a_avr, log_b_avr, T)


        return darkcurrent
        

    def darkcurrent2D(self, T, mode):  # electrons/s
        """Get an 2D field of dark currents for a CCD. 

        Args:
            T (float): Temperature of CCD
            mode (str): Gain mode/ Signal mode for CCD 

        Returns:
            darkcurrent (np.array): average dark current of the CCD

        """

        if mode == 'High':
            log_a_img_avr=self.log_a_img_avr_HSM
            log_b_img_avr=self.log_b_img_avr_HSM
            log_a_img_std=self.log_a_img_err_HSM
            log_b_img_std=self.log_b_img_err_HSM
        elif mode == 'Low':
            log_a_img_avr=self.log_a_img_avr_LSM
            log_b_img_avr=self.log_b_img_avr_LSM 
            log_a_img_std=self.log_a_img_err_LSM
            log_b_img_std=self.log_b_img_err_LSM           
        else:
            raise Exception("Undefined mode")
        darkcurrent=self.getrawdark(log_a_img_avr, log_b_img_avr, T)

        return darkcurrent
        



    def ro_avr(self, mode):
        """?. 

        Args:
            mode (str): Gain mode/ Signal mode for CCD 

        Returns:
            ro_avr ?

        """
        if mode == 'High':
            ro_avr = self.ro_avr_HSM
        elif mode == 'Low':
            ro_avr = self.ro_avr_LSM
        else:
            raise Exception("Undefined mode")
        return ro_avr

    def alpha_avr(self, mode):  # electrons/LSB
        """?. 

        Args:
            mode (str): Gain mode/ Signal mode for CCD 

        Returns:
            alpha_avr ?

        """
        if mode == 'High':
            alpha_avr = self.alpha_avr_HSM
        elif mode == 'Low':
            alpha_avr = self.alpha_avr_LSM
        else:
            raise Exception("Undefined mode")
        return alpha_avr

    def flatfield(self):  
        """

        Returns:
            Flatfield of the CCD as taken in High signal mode. 
            Flatfield is difined to be 1 in on average in the center of the image.
            Generally a merged version between 0 C flatfield without baffle and 20C flatfield with baffle are used.

        """        
        flatfield = self.flatfield_HSM

        return flatfield

    def get_table(self,CCDitem):
        """Check to see if there is a precalculated non linearity table for the CCDMacro used 

        Args:
            CCDitem (dict): Dictionary of type CCDitem

        Returns:
            lookup_table (np.array): a lookup table for the non-linearty

        """
        
        if self.tables is None:
            table = None

        else: 
            df_table = self.tables.loc[(self.tables['CCDSEL'] == CCDitem['CCDSEL']) & (self.tables['NRBIN'] == CCDitem['NRBIN']) & (self.tables['NCBIN CCDColumns'] == CCDitem['NCBIN CCDColumns']) & (self.tables['GAIN Mode'] == CCDitem['GAIN Mode'])]
        
            if len(df_table)>0:
                table = np.load(self.tablefolder + df_table.iloc[0]['Tablefile'] +'.npy')
            else:
                table = None
            
        return table

    def reload_table(self):
        """Reloads non-linearity tables 

        Args:
            None

        Returns:
            None

        """
        self.tables = pd.read_csv(self.tablefolder + 'tables.csv')
        
    def get_channel_quaternion(self):
        """Read the channel quaternion from file
        Args:

        Returns:
            quaternion
        """
        return self.qprime

    def get_artifact_mask(self):
        """Read the artifact masks from file
        Args:

        Returns:
            artifact_masks (dataframe): panda dataframe containing the masks correcting for the artifact in nadir images
        """
        return self.artifact_masks

    def get_single_event(self,CCDitem):
        """Read the artifact masks from file
        Args:
            CCDitem (dict): Dictionary of type CCDitem

        Returns:
            se_mask (np.array): numpy array which marks any single event in image
        """
        df = self.single_event
        date = np.datetime64(CCDitem['EXP Date'],'s').astype(datetime)
        single_events = df[df.datetime==date]
        
        se_mask = np.zeros(CCDitem['IMAGE'].shape)
        for i in range(len(single_events)):
            se_mask[single_events.iloc[i]['Y'],single_events.iloc[i]['X']] = 1

        return se_mask

    def get_hotpixel_map(self,CCDitem):
        """
        Function to get the hotpixel map for a given date

        Arguments
        ----------
        CCDitem : Dict holding information about the image to get hotpixel map for.

        Returns
        -------
        mapdate : datetime item giving the date of the map
            if no valid map this will be the same as the date requested 
            
        hotpixel_map : array[unit16] or empty array if no valid data
            map of hotpixel counts for the given date 
        """

        df = self.hot_pixels
        date = np.datetime64(CCDitem['EXP Date'],'s').astype(datetime)
        channelname = CCDitem["channel"]
        row = df[(df.datetime.dt.date == date.date()) & (df.channel == channelname)]
        if len(row)>0:
            hotpixel_map = row['HPM'].values[0]
            mapdate = row['datetime'].values[0]
        else:
            mapdate = date
            hotpixel_map = np.array([])

        return mapdate, hotpixel_map

class nonLinearity:
    """Class to represent a non-linearity for a MATS CCD.

    This class is used to generalize the non-linearity for a MATS CCD. The non-linearity spline is
    generated using the functions in the database_generation.linearity module.

    Attributes:
        fittype (str): Type of fit the non-linearity is representing
        fit_threshold (float) = max value of measured data used in fitting of the non-linearity
        saturation (float) = true value where the CCD is considered saturated, all values above this are set to the forward model with this value as input.
        non_lin_important (float) = non_lin_important the true value where non_linearity becomes important.
        channel (str) = channel name 
        fit_parameters (list, np.array or obj) = parmeter of object describing the non-linearity fit. 
        covariance (np.array) = covariances of the non-linear fit
    """
    def __init__(self,channel, fittype='threshold2', fit_parameters=None, covariance=None, fit_threshold=1e9,dfdx_saturation=0.05,dfdx_non_lin_important=0.5):
        """Init method for CCD class

        Args:
            channel (str) = channel name 
            fittype (str): Type of fit the non-linearity is representing
            fit_parameters (list, np.array or obj) = parmeter of object describing the non-linearity fit.
            covariance (np.array) = numpy array describing the unceritainty of the fit
            fit_threshold (optional) = max value of measured data used in fitting of the non-linearity
            dfdx_saturation (float) = the first derivative at which the sensor is considered saturated (default 0.05)
            dfdx_non_lin_important (float) = the first derivative value where non_linearity becomes important.
            
        """
        self.fittype = fittype
        self.fit_threshold = fit_threshold
        self.channel = channel

        if isinstance(covariance, np.ndarray):
            self.covariance = covariance
        else:
            self.covariance = None

        if fit_parameters == None:
            self.saturation = 1e9
            self.non_lin_important = 1e9

            if fittype=='polyfit1':
                self.fit_parameters = np.array([1,0])
            elif fittype=='polyfit2':
                self.fit_parameters = np.array([0,1,0])
            elif fittype=='threshold2':
                a,b,e = (1,0,0)
                self.fit_parameters = [a,b,e]
            else:
                raise NotImplementedError
        else:
            self.fit_parameters = fit_parameters
            self.saturation = self.calc_non_lin_important(dfdx_saturation)
            self.non_lin_important = self.calc_non_lin_important(dfdx_non_lin_important)


    def get_measured_image(self, image_true):
        """Method to get a measured value for a given true image (forward model)

        Args:
            x_true (np.array): True image of the signal

        Returns:
            (np.array) measured image taking into account non-linearity. 

        """
        image_measured = np.zeros(image_true.shape)
        if image_measured.ndim == 1:
            for i in range(image_measured.shape[0]):
                image_measured[i] = self.get_measured_value(image_true[i])
        else:
            for i in range(image_measured.shape[0]):
                for j in range(image_measured.shape[1]):
                    image_measured[i,j] = self.get_measured_value(image_true[i,j])

        return image_measured

    def get_measured_value(self,x_true):
        """Method to get a measured value for a given true value (forward model)

        Args:
            x_true (float): True value of the signal

        Returns:
            (float) measured value taking into account non-linearity. 

        """
        # function to get the expected measured count for a given true count value
        # returns value and an errorcode: 
        # 0 = all good, 1 = non linear part important, 2 = value exceeds fit threshold 

        if (self.fittype=='polyfit1') or (self.fittype=='polyfit2'):
            if np.any(x_true > self.saturation):
                return self.saturation
            else:
                return np.polyval(self.fit_parameters,x_true)

        elif self.fittype == 'threshold2':
            a = self.fit_parameters[0]
            b = self.fit_parameters[1]
            e = self.fit_parameters[2]

            if np.any(x_true < e):
                return a*x_true             

            elif np.any(x_true<self.saturation):
                return b*(x_true-e)**2+a*(x_true-e)+a*e

            else:
                return b*(self.saturation-e)**2+a*(self.saturation-e)+a*e

        else:
            raise NotImplementedError(self.fittype)

        
    def get_measured_saturation(self):
        return self.get_measured_value(self.saturation)

    def get_measured_non_lin_important(self):
        return self.get_measured_value(self.non_lin_important)

    def calc_non_lin_important(self,beta):
        if self.fittype != 'threshold2':
            raise NotImplementedError
        else:
            a = self.fit_parameters[0]
            b = self.fit_parameters[1]
            e = self.fit_parameters[2]
            if b<0:
                threshold = (beta+2*b*e-a)/(2*b)
            else:
                beta = 1+beta
                threshold = (beta+2*b*e-a)/(2*b)
        
        return threshold

    def get_random_fit_parameter(self):
        """Method to returns the fit parameters, except that they are randomly sampled from the fitted distribution (rather than just the mean)

        Args:

        Returns:
            (np.array) fit parameters from random sample 

        """
        if isinstance(self.covariance, np.ndarray):
            return np.random.multivariate_normal(self.fit_parameters, self.covariance, size=None, check_valid='warn', tol=1e-8)
        else:
            return self.fit_parameters
            


class Instrument:
    """Class to hold a set of MATS CCDs

    This class is made to have a object to send around containing all the information about the MATS instrument 
    calibration. 

    Attributes:
        IR1 (CCD): MATS CCD channel
        IR2: (CCD): MATS CCD channel
        IR3: (CCD): MATS CCD channel
        IR4: (CCD): MATS CCD channel
        UV1: (CCD): MATS CCD channel
        UV2: (CCD): MATS CCD channel
        NADIR: (CCD): MATS CCD channel
        KTH_test_channel (CCD): MATS CCD channel
        calibration_file: string containing the info in the calibration file
        """

    def __init__(self, calibration_file, channel=None, start_datetime: datetime = datetime(2000,1,1), end_datetime: datetime = datetime(3000,1,1)):

        """Init method for Instrument class

        Args:
            calibration_file (str): calibration file containing paths to calibration data
            channel (str, list, optional): name of channel, a list of channel names or empty which creates a oject with the 6 standard channels.            
            start_datetime (datetime): datetime on which start extraction calibration parameters (default: None) 
            end_datetime (datetime): datetime on which to end extraction calibration parameters (default: None) 

        """

        f = open(calibration_file)
        self.calibration_file=f.read()
        f.close()

        self.IR1 = None
        self.IR2 = None
        self.IR3 = None
        self.IR4 = None
        self.UV1 = None
        self.UV2 = None
        self.NADIR = None
        self.KTH_test_channel = None
        
        if channel == None:
            self.IR1 = CCD("IR1",calibration_file,start_datetime,end_datetime)
            self.IR2 = CCD("IR2",calibration_file,start_datetime,end_datetime)
            self.IR3 = CCD("IR3",calibration_file,start_datetime,end_datetime)
            self.IR4 = CCD("IR4",calibration_file,start_datetime,end_datetime)
            self.UV1 = CCD("UV1",calibration_file,start_datetime,end_datetime)
            self.UV2 = CCD("UV2",calibration_file,start_datetime,end_datetime)
            self.NADIR = CCD("NADIR",calibration_file,start_datetime,end_datetime)
            
        elif type(channel) == str:
            setattr(self, channel, CCD("channel",calibration_file))
        elif len(channel)>0:
            for channnels in enumerate(channel):
                setattr(self, channnels, CCD(channnels,calibration_file))

    def get_CCD(self,channel):
        """Method to get a CCD from the instrument

        Args:
            channel (str): name of channel

        Returns:
            (CCD) MATS CCD object a.k.a CCDunit

        """
        return getattr(self,channel)
     

class Photometer:
    """
    Class to represent the two photometers on MATS

    The class contains attributes and methods for extracting calibration parameters for the
    photometers.

    Attributes:
        cal_therm (dict): calibration table Thermistors
        cal_rad (dict): calibration table Photometers
    """

    def __init__(self, calibration_file: str):
        """Init method for Photometer class

        Args:
            calibration_file (str): calibration file containing paths to calibration data
        """
        #   import matlab .mat calibration files into dicts
        calibration_data = toml.load(calibration_file)
        self.cal_therm = loadmat(calibration_data["photometer"]["thermistor_table"]) # Thermistors

        #read in splines for photometer calibration
        with open(calibration_data["photometer"]["FM1_spline"], 'rb') as fp:
            self.cal_rad_FM1 = pickle.load(fp)
        with open(calibration_data["photometer"]["FM2_spline"], 'rb') as fp:
            self.cal_rad_FM2 = pickle.load(fp)
