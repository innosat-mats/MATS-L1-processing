#%%
import json
import numpy as np
from matplotlib import pyplot as plt
from mats_l1_processing.L1_calibration_functions import CCD
from mats_l1_processing.L1_calibration_functions import inverse_model_real,total_model_scalar,check_true_value_max,test_for_saturation
from mats_l1_processing.read_in_functions import channel_num_to_str, add_and_rename_CCDitem_info
import pandas as pd
import operator as op
# #%%
# def gen_non_linear_table(CCDitem,calibrationfile,covariance_exp=None,covariance_col=None,type='inverse'):

#     #Add stuff not in the non linear tables (but required for add_and_rename)
#     CCDitem["read_from"] = 'rac'
#     CCDitem["EXP Nanoseconds"] = 0
#     CCDitem["BC"] = '[]'

#     #CCDitem["channel"] = channel_num_to_str(CCDitem['CCDSEL'])
#     #CCDitem["RID"] = 'CCD' + str(CCDitem['CCDSEL'])
#     #CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)
#     CCDitem = add_and_rename_CCDitem_info(CCDitem)
#     CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)

#     if type(covariance_exp) != type(None):
#         default_parameters = CCDitem["CCDunit"].non_linearity_pixel.fit_parameters
#         CCDitem["CCDunit"].non_linearity_pixel.fit_parameters = [default_parameters[0]+covariance_exp['a'],default_parameters[1]+covariance_exp['b'],default_parameters[2]+covariance_exp['e']]
#         CCDitem["CCDunit"].non_linearity_pixel.saturation = CCDitem["CCDunit"].non_linearity_pixel.calc_non_lin_important(0.01)
#         CCDitem["CCDunit"].non_linearity_pixel.non_lin_important = CCDitem["CCDunit"].non_linearity_pixel.calc_non_lin_important(0.01)


#         default_parameters = CCDitem["CCDunit"].non_linearity_sumwell.fit_parameters
#         CCDitem["CCDunit"].non_linearity_sumwell.fit_parameters = [default_parameters[0]+covariance_col['a'],default_parameters[1]+covariance_col['b'],default_parameters[2]+covariance_col['e']]
#         CCDitem["CCDunit"].non_linearity_sumwell.saturation = CCDitem["CCDunit"].non_linearity_sumwell.calc_non_lin_important(0.01)
#         CCDitem["CCDunit"].non_linearity_sumwell.non_lin_important = CCDitem["CCDunit"].non_linearity_sumwell.calc_non_lin_important(0.01)

#     x_measured = np.arange(0,2**16-1,1)
#     x_true = np.zeros(x_measured.shape)
#     flag = np.zeros(x_measured.shape)
    
#     for i in range(len(x_measured)):
#         x_true[i],flag[i] = inverse_model_real(CCDitem,x_measured[i],method='Nelder-Mead')

#     return x_true,x_measured,CCDitem,flag

#%%

def gen_non_linear_table(CCDitem,calibrationfile,covariance_exp=None,covariance_col=None,fittype='inverse',randomize=False):

    #Add stuff not in the non linear tables (but required for add_and_rename)
    CCDitem["read_from"] = 'rac'
    CCDitem["EXP Nanoseconds"] = 0
    CCDitem["BC"] = '[]'

    #CCDitem["channel"] = channel_num_to_str(CCDitem['CCDSEL'])
    #CCDitem["RID"] = 'CCD' + str(CCDitem['CCDSEL'])
    #CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)
    CCDitem = add_and_rename_CCDitem_info(CCDitem)
    CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)
    
    if randomize:
        CCDitem["CCDunit"].non_linearity_pixel.fit_parameters = CCDitem["CCDunit"].non_linearity_pixel.get_random_fit_parameter()
        CCDitem["CCDunit"].non_linearity_sumwell.fit_parameters = CCDitem["CCDunit"].non_linearity_sumwell.get_random_fit_parameter()

    if fittype == 'interp':
        CCDitem["CCDunit"].non_linearity_pixel.saturation = CCDitem["CCDunit"].non_linearity_pixel.calc_non_lin_important(0.01)
        CCDitem["CCDunit"].non_linearity_sumrow.saturation = CCDitem["CCDunit"].non_linearity_sumrow.calc_non_lin_important(0.01)
        CCDitem["CCDunit"].non_linearity_sumwell.saturation = CCDitem["CCDunit"].non_linearity_sumwell.calc_non_lin_important(0.01)

    if type(covariance_exp) != type(None):

        default_parameters = CCDitem["CCDunit"].non_linearity_pixel.fit_parameters
        CCDitem["CCDunit"].non_linearity_pixel.fit_parameters = [default_parameters[0]+covariance_exp['a'],default_parameters[1]+covariance_exp['b'],default_parameters[2]+covariance_exp['e']]
        CCDitem["CCDunit"].non_linearity_pixel.saturation = CCDitem["CCDunit"].non_linearity_pixel.calc_non_lin_important(0.01)
        CCDitem["CCDunit"].non_linearity_pixel.non_lin_important = CCDitem["CCDunit"].non_linearity_pixel.calc_non_lin_important(0.01)

        default_parameters = CCDitem["CCDunit"].non_linearity_sumwell.fit_parameters
        CCDitem["CCDunit"].non_linearity_sumwell.fit_parameters = [default_parameters[0]+covariance_col['a'],default_parameters[1]+covariance_col['b'],default_parameters[2]+covariance_col['e']]
        CCDitem["CCDunit"].non_linearity_sumwell.saturation = CCDitem["CCDunit"].non_linearity_sumwell.calc_non_lin_important(0.01)
        CCDitem["CCDunit"].non_linearity_sumwell.non_lin_important = CCDitem["CCDunit"].non_linearity_sumwell.calc_non_lin_important(0.01)

    try:
        CCDunit = CCDitem["CCDunit"]
    except:
        raise Exception("No CCDunit defined for the CCDitem")

    nrowbin = CCDitem["NRBIN"]
    ncolbin = CCDitem["NColBinCCD"]

    if fittype=='interp':
        x_true_samples = np.arange(0,2**17-1,1)
        x_measured_sample = np.zeros(x_true_samples.shape)

        for i in range(len(x_true_samples)):
            x_measured_sample[i] = total_model_scalar(x_true_samples[i],CCDunit,nrowbin,ncolbin)

        x_measured = np.arange(0,2**16-1,1)
        x_true = np.interp(x_measured,x_measured_sample,x_true_samples)
        flag = np.zeros(x_true.shape)

        for i in range(len(x_true)):
            flag[i], x_sat = test_for_saturation(CCDunit,nrowbin,ncolbin,x_measured[i])
            if (flag[i] == 3):
                x_true[i] = x_sat

            flag[i],x_true[i] = check_true_value_max(CCDunit,nrowbin,ncolbin,x_true[i],flag[i])

    elif fittype=='inverse':
        x_measured = np.arange(0,2**16-1,1)
        x_true = np.zeros(x_measured.shape)
        flag = np.zeros(x_measured.shape)
        
        for i in range(len(x_measured)):
            x_true[i],flag[i] = inverse_model_real(CCDitem,x_measured[i],method='Nelder-Mead')

    else:
        raise ValueError('fittype need to be inverse or interp')

    return x_true,x_measured,CCDitem,flag

#%%
def get_covariances(channel):
    df = pd.read_csv('calibration_data/linearity/final/covariance_col')
    cov_col = df.iloc[channel-1]

    df = pd.read_csv('calibration_data/linearity/final/covariance_row')
    cov_row = df.iloc[channel-1]

    df = pd.read_csv('calibration_data/linearity/final/covariance_exp')
    cov_exp = df.iloc[channel-1]

    return cov_col,cov_row,cov_exp

def convert_covariances(covariance,sign1,sign2,sign3):
    covariance[0] = sign1*(covariance[0])
    covariance[1] = sign2*(covariance[1])
    covariance[2] = sign3*(covariance[2])

    return covariance

def gen_non_linear_error(CCDitem,calibrationfile):

    #Add stuff not in the non linear tables (but required for add_and_rename)
    CCDitem["read_from"] = 'rac'
    CCDitem["EXP Nanoseconds"] = 0
    CCDitem["BC"] = '[]'

    CCDitem = add_and_rename_CCDitem_info(CCDitem)
    CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)
  
    covariance_exp,covariance_row,covariance_col = get_covariances(CCDitem['CCDSEL'])

    return covariance_exp,covariance_row,covariance_col

def covariance_test(covariance_exp,covariance_col,items,i,name,fittype='inverse'):

    map = {"p":1,"n":-1}
    convert_covariances(covariance_exp,map[name[0]],map[name[1]],map[name[2]])
    convert_covariances(covariance_col,map[name[0]],map[name[1]],map[name[2]])

    x_true_2,x_measured_2,CCDitem_2,flag_2 = gen_non_linear_table(items[i],'calibration_data/calibration_data.toml',covariance_exp,covariance_col,fittype)
    table = np.array([x_true_2,flag_2,x_measured_2])
    np.save(str(i)+name+ '.npy',table)


directory = "calibration_data/linearity/tables/20220802/"
df = pd.read_csv(directory + "tables.csv")
items = df.to_dict("records")


# %%
# names = ['ppp','ppn','pnp','npp','pnn','nnp','npn','nnn']

# #%%
# for i in range(0,len(items)):
   
#    x_true_2,x_measured_2,CCDitem_2,flag_2 = gen_non_linear_table(items[i],'calibration_data/calibration_data.toml',fittype='interp')
#    table = np.array([x_true_2,flag_2,x_measured_2])
#    np.save(str(i) + '.npy',table)
#    covariance_exp,_,covariance_col = gen_non_linear_error(items[i],'calibration_data/calibration_data.toml')
#    for j in range(len(names)):
#        covariance_test(covariance_exp,covariance_col,items,i,names[j],fittype='interp')

#%%

# for i in range(len(items)):
#     non_lins = []
#     for j in range(len(names)):
#         non_lins.append(np.load(str(i) + names[j] + '.npy'))    
#         plt.plot(non_lins[j][0,:],non_lins[j][2,:])
    
#     non_lins.append(np.load(str(i) + '.npy'))

#     non_lin_important = [ n for n,i in enumerate(non_lins[-1][1,:]) if i==1 ][0]
    
#     plt.plot(non_lins[-1][0,:],non_lins[-1][2,:],linewidth=4)
#     plt.plot([0,non_lin_important*1.5],[non_lin_important,non_lin_important])
#     plt.ylim([0,non_lin_important*1.5])
#     plt.xlim([0,non_lin_important*1.5])
#     plt.title('test' + str(i+1))
#     plt.xlabel('true counts')
#     plt.ylabel('measured counts')
#     plt.savefig('test' + str(i+1),dpi=600)
#     plt.show()

#     array = np.array(non_lins)

#     # plt.plot(array[:,0,1:].std(0)/non_lins[-1][0,1:])
#     # plt.plot([array[-1,0,non_lin_important],array[-1,0,non_lin_important]],[0,1])
#     # plt.ylim([0,1])
#     # plt.xlim([0,non_lin_important*1.5])
#     # plt.title('test' + str(i+1))
#     # plt.xlabel('standard_deviation_of_true_counts')
#     # plt.ylabel('measured counts')
#     # plt.savefig('test' + str(i+1),dpi=600)
#     # plt.show()

# %%
# names = ['./','tmp/']

# for i in range(len(items)):
#     non_lins = []
#     for j in range(len(names)):
#         non_lins.append(np.load(names[j] + str(i) + '.npy'))    
#         plt.plot(non_lins[j][0,:],non_lins[j][2,:])
    
#     non_lin_important = [ n for n,i in enumerate(non_lins[-1][1,:]) if i==1 ][0]
    
#     #plt.plot([0,non_lin_important*1.5],[non_lin_important,non_lin_important])
#     #plt.ylim([0,non_lin_important*1.5])
#     #plt.xlim([0,non_lin_important*1.5])
#     plt.title('test' + str(i+1))
#     plt.xlabel('true counts')
#     plt.ylabel('measured counts')
#     plt.savefig('test' + str(i+1),dpi=600)
#     plt.show()
# %%

n_samples = 10

# for i in range(len(items)):
#     for n in range(n_samples):
#         x_true_2,x_measured_2,CCDitem_2,flag_2 = gen_non_linear_table(items[i],'calibration_data/calibration_data.toml',fittype='interp',randomize=True)
#         table = np.array([x_true_2,flag_2,x_measured_2])
#         np.save(str(i) + '_' + str(n) + '_' + '.npy',table)

# i = 0
for i in range(len(items)):
    for n in range(n_samples):
        table = np.load(str(i) + '_' + str(n) + '_' + '.npy')
        plt.plot(table[0,:],table[2,:])
        plt.ylim([0,10000])
    plt.show()
# %%
