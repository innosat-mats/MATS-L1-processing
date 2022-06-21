import numpy as np

class nonLinearity:
    def __init__(self, fittype, threshold,non_lin_important,channel,poly_or_spline):
        self.fittype = fittype
        self.threshold = threshold
        self.non_lin_important = non_lin_important
        self.channel = channel
        self.poly_or_spline = poly_or_spline


    def get_measured_value(self,x_true):
        # function to get the expected measured count for a given true count value
        # returns value and an errorcode: 
        # 0 = all good, 1 = non linear part important, 2 = value exceeds fit threshold 

        if (self.fittype=='polyfit1') or (self.fittype=='polyfit2'):
            if x_true > self.threshold:
                return self.threshold
            else:
                return np.polyval(self.poly_or_spline,x_true)

        elif self.fittype == 'threshold2':
            a = self.poly_or_spline[0]
            b = self.poly_or_spline[1]
            e = self.poly_or_spline[2]

            if x_true < e:
                return a*x_true             

            elif x_true<self.threshold:
                return b*(x_true-e)**2+a*x_true+a*e

            else:
                return self.threshold

        else:
            raise NotImplementedError
