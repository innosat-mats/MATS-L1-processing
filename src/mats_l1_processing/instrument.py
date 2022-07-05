# -*- coding: utf-8 -*-
"""Classes containing information about the MATS instrument.

This module has a CCD class which contains the information about a specific CCD 
in MATS such as dark-current patterns, non-linearity response and flat fields. Another
class is used as a container for several CCDs. Finally a class for a generalized non-linearity
is included. 

"""

import toml
import numpy as np
import scipy
import pickle

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
        non_linearity_pixel (nonLinearity): Linearity of the shift register
        non_linearity_pixel (nonLinearity): Linerarity of the summation well

        ampcorrection (float): correction for pre-amplification

    """

    def __init__(self, channel, calibration_file):

        """Init method for CCD class

        Args:
            channel (str): Name of channel
            calibration_file (str): calibration file containing paths to calibration files

        """

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

        filename = (
            calibration_data["darkcurrent"]["folder"]
            + "FM0"
            + str(CCDID)
            + "_CCD_DC_calibration.mat"
        )

        mat = scipy.io.loadmat(filename)

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

        # 1D dark current subtraction stuff
        self.log_a_avr_HSM = mat["log_a_avr_HSM"]
        self.log_a_std_HSM = mat["log_a_std_HSM"]
        self.log_b_avr_HSM = mat["log_b_avr_HSM"]
        self.log_b_std_HSM = mat["log_b_std_HSM"]

        self.log_a_avr_LSM = mat["log_a_avr_LSM"]
        self.log_a_std_LSM = mat["log_a_std_LSM"]
        self.log_b_avr_LSM = mat["log_b_avr_LSM"]
        self.log_b_std_LSM = mat["log_b_std_LSM"]

        # 2D dark current subtraction stuff
        self.log_a_img_avr_LSM = mat["log_a_img_avr_LSM"]
        self.log_a_img_err_LSM = mat["log_a_img_err_LSM"]
        self.log_b_img_avr_LSM = mat["log_b_img_avr_LSM"]
        self.log_b_img_err_LSM = mat["log_b_img_err_LSM"]

        self.log_a_img_avr_HSM = mat["log_a_img_avr_HSM"]
        self.log_a_img_err_HSM = mat["log_a_img_err_HSM"]
        self.log_b_img_avr_HSM = mat["log_b_img_avr_HSM"]
        self.log_b_img_err_HSM = mat["log_b_img_err_HSM"]

        # Flatfields
        self.flatfield_HSM = np.load(
            calibration_data["flatfield"]["flatfieldfolder"]
            + "flatfield_"
            + channel
            + "_HSM.npy"
        )
        self.flatfield_LSM = np.load(
            calibration_data["flatfield"]["flatfieldfolder"]
            + "flatfield_"
            + channel
            + "_LSM.npy"
        )

        # Non-linearity
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

        # Amplification correction for UV channels
        if self.channel == "UV1" or self.channel == "UV2":
            self.ampcorrection = (
                3 / 2
            )
        else:
            self.ampcorrection = 1

    def darkcurrent(self, T, mode):  # electrons/s
        """Get an average dark current for a channel. 

        This method can be used of the dark current signal is very low, since using the 2D functions become noisy then.

        Args:
            T (float): Temperature of CCD
            mode (str): Gain mode/ Signal mode for CCD 

        Returns:
            darkcurrent(float): average dark current of the CCD

        """

        if mode == "high":
            darkcurrent = 10 ** (self.log_a_avr_HSM * T + self.log_b_avr_HSM)
        elif mode == "low":
            darkcurrent = 10 ** (self.log_a_avr_LSM * T + self.log_b_avr_LSM)
        else:
            raise ValueError('Mode must be "high" or "low"')
        return darkcurrent

    def darkcurrent2D(self, T, mode):  # electrons/s
        """Get an 2D field of dark currents for a CCD. 

        Args:
            T (float): Temperature of CCD
            mode (str): Gain mode/ Signal mode for CCD 

        Returns:
            darkcurrent (np.array): average dark current of the CCD

        """
        if mode == 0:
            darkcurrent = 10 ** (self.log_a_img_avr_HSM * T + self.log_b_img_avr_HSM)
        elif mode == 1:
            darkcurrent = 10 ** (self.log_a_img_avr_LSM * T + self.log_b_img_avr_LSM)
        else:
            print("Undefined mode")
        return darkcurrent

    def ro_avr(self, mode):
        """?. 

        Args:
            mode (str): Gain mode/ Signal mode for CCD 

        Returns:
            ro_avr ?

        """
        if mode == 0:
            ro_avr = self.ro_avr_HSM
        elif mode == 1:
            ro_avr = self.ro_avr_LSM
        else:
            print("Undefined mode")
        return ro_avr

    def alpha_avr(self, mode):  # electrons/LSB
        """?. 

        Args:
            mode (str): Gain mode/ Signal mode for CCD 

        Returns:
            alpha_avr ?

        """
        if mode == 0:
            alpha_avr = self.alpha_avr_HSM
        elif mode == 1:
            alpha_avr = self.alpha_avr_LSM
        else:
            print("Undefined mode")
        return alpha_avr

    def flatfield(self, mode):  # return flatfield at 0C for the CCD (in LSB?)
        """?. 

        Args:
            mode (str): Gain mode/ Signal mode for CCD 

        Returns:
            flatfield ?

        """
        if mode == 0:
            flatfield = self.flatfield_HSM
        elif mode == 1:
            flatfield = self.flatfield_LSM
        else:
            print("Undefined mode")

        return flatfield



class nonLinearity:
    """Class to represent a non-linearity for a MATS CCD.

    This class is used to generalize the non-linearity for a MATS CCD. The non-linearity spline is
    generated using the functions in the database_generation.linearity module.

    Attributes:
        fittype (str): Type of fit the non-linearity is representing
        fit_threshold (float) = max value of measured data used in fitting of the non-linearity
        saturation (float) = measured value where the CCD is considered saturated, all values above this are set to this value.
        non_lin_important (float) = non_lin_important the true value where non_linearity becomes important.
        channel (str) = channel name 
        fit_parameters (list, np.array or obj) = parmeter of object describing the non-linearity fit. 

    """
    def __init__(self,channel, fittype, fit_parameters=None, fit_threshold=1e9,saturation=1e9,non_lin_important=1e9):
        """Init method for CCD class

        Args:
            channel (str) = channel name 
            fittype (str): Type of fit the non-linearity is representing
            fit_threshold (optional) = max value of measured data used in fitting of the non-linearity
            saturation (float) = measured value where the CCD is considered saturated, all values above this are set to this value.
            non_lin_important (float) = non_lin_important the true value where non_linearity becomes important.
            fit_parameters (list, np.array or obj) = parmeter of object describing the non-linearity fit.
            
        """
        self.fittype = fittype
        self.fit_threshold = fit_threshold
        self.saturation = saturation
        self.non_lin_important = non_lin_important #the true value where non_linearity becomes important
        self.channel = channel
        if fit_parameters == None:
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

    def get_measured_image(self, image_true):
        """Method to get a measured value for a given true image (forward model)

        Args:
            x_true (np.array): True image of the signal

        Returns:
            (np.array) measured image taking into account non-linearity. 

        """
        image_measured = np.zeros(image_true.shape)
        for i in range(image_measured.shape[0]):
            for j in range(image_measured.shape[1]):
                image_measured[i,j] = self.get_measured_value(image_true[i,j])

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
            if x_true > self.non_lin_important:
                return self.threshold
            else:
                return np.polyval(self.fit_parameters,x_true)

        elif self.fittype == 'threshold2':
            a = self.fit_parameters[0]
            b = self.fit_parameters[1]
            e = self.fit_parameters[2]

            if x_true < e:
                return a*x_true             

            elif x_true<self.non_lin_important:
                return b*(x_true-e)**2+a*(x_true-e)+a*e

            else:
                return b*(self.non_lin_important-e)**2+a*(self.non_lin_important-e)+a*e

        else:
            raise NotImplementedError

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

        """

    def __init__(self, calibration_file, channel=None):

        """Init method for CCD class

        Args:
            calibration_file (str): calibration file containing paths to calibration data
            channel (str, list, optional): name of channel, a list of channel names or empty which creates a oject with the 6 standard channels.            

        """

        self.IR1 = None
        self.IR2 = None
        self.IR3 = None
        self.IR4 = None
        self.UV1 = None
        self.UV2 = None
        self.NADIR = None
        self.KTH_test_channel = None
        
        if channel == None:
            self.IR1 = CCD("IR1",calibration_file)
            self.IR2 = CCD("IR2",calibration_file)
            self.IR3 = CCD("IR3",calibration_file)
            self.IR4 = CCD("IR4",calibration_file)
            self.UV1 = CCD("UV1",calibration_file)
            self.UV2 = CCD("UV2",calibration_file)
            
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
        getattr(self,channel)
