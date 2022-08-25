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

def gen_non_linear_table(CCDitem,calibrationfile,fittype='inverse',randomize=False):

    #Add stuff not in the non linear tables (but required for add_and_rename)
    CCDitem["read_from"] = 'rac'
    CCDitem["EXP Nanoseconds"] = 0
    CCDitem["BC"] = '[]'

    #CCDitem["channel"] = channel_num_to_str(CCDitem['CCDSEL'])
    #CCDitem["RID"] = 'CCD' + str(CCDitem['CCDSEL'])
    #CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)

    CCDitem = add_and_rename_CCDitem_info(CCDitem)
    print(CCDitem["channel"])
    CCDitem["CCDunit"] = CCD(CCDitem["channel"], calibrationfile)
    
    CCDitem["CCDunit"].non_linearity_pixel.non_lin_important = CCDitem["CCDunit"].non_linearity_pixel.calc_non_lin_important(0.5)
    CCDitem["CCDunit"].non_linearity_sumrow.non_lin_important = CCDitem["CCDunit"].non_linearity_sumrow.calc_non_lin_important(0.5)
    CCDitem["CCDunit"].non_linearity_sumwell.non_lin_important = CCDitem["CCDunit"].non_linearity_sumwell.calc_non_lin_important(0.5)


    if randomize:
        CCDitem["CCDunit"].non_linearity_pixel.fit_parameters = CCDitem["CCDunit"].non_linearity_pixel.get_random_fit_parameter()
        CCDitem["CCDunit"].non_linearity_sumwell.fit_parameters = CCDitem["CCDunit"].non_linearity_sumwell.get_random_fit_parameter()

        CCDitem["CCDunit"].non_linearity_pixel.saturation = CCDitem["CCDunit"].non_linearity_pixel.calc_non_lin_important(0.05)
        CCDitem["CCDunit"].non_linearity_sumrow.saturation = CCDitem["CCDunit"].non_linearity_sumrow.calc_non_lin_important(0.05)
        CCDitem["CCDunit"].non_linearity_sumwell.saturation = CCDitem["CCDunit"].non_linearity_sumwell.calc_non_lin_important(0.05)

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

directory = "calibration_data/linearity/tables/20220802/"
df = pd.read_csv(directory + "tables.csv")
items = df.to_dict("records")

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

n_samples = 5
# for i in range(len(items)):
#     x_true_2,x_measured_2,CCDitem_2,flag_2 = gen_non_linear_table(items[i],'calibration_data/calibration_data.toml',fittype='interp')
#     table = np.array([x_true_2,flag_2,x_measured_2])
#     np.save(str(i) + '.npy',table)
#     for n in range(n_samples):
#         x_true_2,x_measured_2,CCDitem_2,flag_2 = gen_non_linear_table(items[i],'calibration_data/calibration_data.toml',fittype='interp',randomize=True)
#         table = np.array([x_true_2,flag_2,x_measured_2])
#         np.save(str(i) + '_' + str(n) + '.npy',table)

# %%

non_lin_fit = []
non_lin_std = []
lin_max_list = []
for i in range(len(items)):
    non_lins = []
    for n in range(n_samples):
        non_lins.append(np.load(str(i) + '_' + str(n) + '.npy'))

        #plt.plot(table[0,:],table[2,:])
        #plt.ylim([0,10000])
    
    non_lins.append(np.load(str(i) + '.npy'))
    lin_max = [ n for n,i in enumerate(non_lins[-1][1,:]) if i==1 ][0]
    lin_max_list.append(lin_max)

    A = np.array(non_lins)
    # plt.plot(A[:,0,:].T)
    # plt.plot([lin_max,lin_max],[0,A[:,0,:].max()],'k--')
    # plt.xlim([0,lin_max*1.5])
    # plt.grid()
    # plt.gca().set_aspect("equal")
    # plt.savefig('test' + str(i+1),dpi=600)
    # plt.show()

    A_diff = A[:-1,0,:]-A[-1,0,:]
    # plt.plot(A_diff[:,:].T)
    # plt.plot([lin_max,lin_max],[A_diff.min(),A_diff.max()],'k--')
    # plt.xlim([0,lin_max*1.5])
    # plt.savefig('test' + str(i+1) + "_abs",dpi=600)
    # plt.show()

    A_std = np.std(A_diff,axis=0)

    # plt.plot(A_std/A[-1,0,:]*100)
    # plt.plot([lin_max,lin_max],[0,A_std.max()],'k--')
    # plt.xlim([0,lin_max*1.5])
    # plt.ylim([0,5])
    # plt.savefig('test' + str(i+1) + "_rel",dpi=600)
    # plt.show()

    non_lin_fit.append(A[-1,0,:])
    non_lin_std.append(A_std)

non_lin_fit = np.array(non_lin_fit)
non_lin_std = np.array(non_lin_std)
lin_max_list = np.array(lin_max_list)

# %%

cmap = plt.get_cmap("tab10")
for i in range(0,6):
    plt.plot(np.arange(0,lin_max_list[i]),non_lin_std[i,:lin_max_list[i]]/non_lin_fit[i,:lin_max_list[i]],color=cmap(items[i]['CCDSEL']),label='channel ' + str(items[i]['CCDSEL']))
    plt.plot(np.arange(lin_max_list[i]+1,len(non_lin_std[i,:])),non_lin_std[i,lin_max_list[i]:-1]/non_lin_fit[i,lin_max_list[i]:-1],color=cmap(items[i]['CCDSEL']),linestyle='dotted',label=None)
plt.title('Error for fullframe macro')
plt.xlabel('measured counts')
plt.ylabel('relative uncertainty')
plt.xlim([0,8e3])
plt.ylim([0,0.1])
plt.legend()
plt.grid()
plt.savefig('error_fullframe',dpi=600)
plt.show()

# # %%
for i in range(6,12):
    plt.plot(np.arange(0,lin_max_list[i]),non_lin_std[i,:lin_max_list[i]]/non_lin_fit[i,:lin_max_list[i]],color=cmap(items[i]['CCDSEL']))
    plt.plot(np.arange(lin_max_list[i]+1,len(non_lin_std[i,:])),non_lin_std[i,lin_max_list[i]:-1]/non_lin_fit[i,lin_max_list[i]:-1],color=cmap(items[i]['CCDSEL']),linestyle='dotted')
plt.title('Error for highres_IR macro')
plt.xlabel('measured counts')
plt.ylabel('relative uncertainty')
plt.xlim([0,64e3])
plt.ylim([0,0.1])
plt.grid()
plt.savefig('error_highres_IR',dpi=600)
plt.show()

# #%%
for i in range(12,18):
    plt.plot(np.arange(0,lin_max_list[i]),non_lin_std[i,:lin_max_list[i]]/non_lin_fit[i,:lin_max_list[i]],color=cmap(items[i]['CCDSEL']))
    plt.plot(np.arange(lin_max_list[i]+1,len(non_lin_std[i,:])),non_lin_std[i,lin_max_list[i]:-1]/non_lin_fit[i,lin_max_list[i]:-1],color=cmap(items[i]['CCDSEL']),linestyle='dotted')
plt.title('Error for lowpixel macro')
plt.xlabel('measured counts')
plt.ylabel('relative uncertainty')
plt.xlim([0,64e3])
plt.ylim([0,0.1])
plt.grid()
plt.savefig('error_lowpixel',dpi=600)
plt.show()

#%%
for i in range(18,24):
    plt.plot(np.arange(0,lin_max_list[i]),non_lin_std[i,:lin_max_list[i]]/non_lin_fit[i,:lin_max_list[i]],color=cmap(items[i]['CCDSEL']))
    plt.plot(np.arange(lin_max_list[i]+1,len(non_lin_std[i,:])),non_lin_std[i,lin_max_list[i]:-1]/non_lin_fit[i,lin_max_list[i]:-1],color=cmap(items[i]['CCDSEL']),linestyle='dotted')
plt.title('Error for highres_UV macro')
plt.xlabel('measured counts')
plt.ylabel('relative uncertainty')
plt.xlim([0,64e3])
plt.ylim([0,0.1])
plt.grid()
plt.savefig('error_highres_UV',dpi=600)
plt.show()

# %%
