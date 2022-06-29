import numpy as np

class nonLinearity:
    def __init__(self, fittype, fit_threshold,non_lin_important,channel,poly_or_spline):
        self.fittype = fittype
        self.fit_threshold = fit_threshold
        self.non_lin_important = non_lin_important #the true value where non_linearity becomes important
        self.channel = channel
        self.poly_or_spline = poly_or_spline


    def get_measured_value(self,x_true):
        # function to get the expected measured count for a given true count value
        # returns value and an errorcode: 
        # 0 = all good, 1 = non linear part important, 2 = value exceeds fit threshold 

        if (self.fittype=='polyfit1') or (self.fittype=='polyfit2'):
            if x_true > self.non_lin_important:
                return self.threshold
            else:
                return np.polyval(self.poly_or_spline,x_true)

        elif self.fittype == 'threshold2':
            a = self.poly_or_spline[0]
            b = self.poly_or_spline[1]
            e = self.poly_or_spline[2]

            if x_true < e:
                return a*x_true             

            elif x_true<self.non_lin_important:
                return b*(x_true-e)**2+a*(x_true-e)+a*e

            else:
                return b*(self.non_lin_important-e)**2+a*(self.non_lin_important-e)+a*e

        else:
            raise NotImplementedError
