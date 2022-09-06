#%%
from cProfile import label
import json
import numpy as np
from matplotlib import pyplot as plt
from mats_l1_processing.L1_calibration_functions import CCD
from mats_l1_processing.L1_calibration_functions import inverse_model_real,total_model_scalar,check_true_value_max,test_for_saturation
from mats_l1_processing.read_in_functions import channel_num_to_str, add_and_rename_CCDitem_info
import pandas as pd
from joblib import Parallel, delayed


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

n_samples = 20

# def loop_over_samples(i,item):
#     print(str(i))
#     x_true_2,x_measured_2,CCDitem_2,flag_2 = gen_non_linear_table(item,'calibration_data/calibration_data.toml',fittype='interp')
#     table = np.array([x_true_2,flag_2,x_measured_2])
#     np.save(str(i) + '.npy',table)
#     for n in range(n_samples):
#         x_true_2,x_measured_2,CCDitem_2,flag_2 = gen_non_linear_table(item,'calibration_data/calibration_data.toml',fittype='interp',randomize=True)
#         table = np.array([x_true_2,flag_2,x_measured_2])
#         np.save(str(i) + '_' + str(n) + '.npy',table)


# Parallel(n_jobs=8)(delayed(loop_over_samples)(i,items[i]) for i in range(len(items)))


# %%

non_lin_fit = []
non_lin_std = []
lin_max_list = []
for i in range(len(items)):
    non_lins = []
    for n in range(n_samples):
        non_lins.append(np.load(str(i) + '_' + str(n) + '.npy'))
    
    non_lins.append(np.load(str(i) + '.npy'))
    lin_max = [ n for n,i in enumerate(non_lins[-1][1,:]) if i==1 ][0]

    A = np.array(non_lins)
    A_diff = A[:-1,0,:]-A[-1,0,:]
    A_std = np.std(A_diff,axis=0)

    non_lin_fit.append(A[-1,0,:])
    non_lin_std.append(A_std)
    lin_max_list.append(lin_max)

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
plot_order = np.array([8,10,11,9,6,7])
for i in plot_order:
    plt.plot(np.arange(0,lin_max_list[i]),non_lin_std[i,:lin_max_list[i]]/non_lin_fit[i,:lin_max_list[i]],color=cmap(items[i]['CCDSEL']),label='channel ' + str(items[i]['CCDSEL']))
    plt.plot(np.arange(lin_max_list[i]+1,len(non_lin_std[i,:])),non_lin_std[i,lin_max_list[i]:-1]/non_lin_fit[i,lin_max_list[i]:-1],color=cmap(items[i]['CCDSEL']),linestyle='dotted',label=None)
plt.title('Error for highres_IR macro')
plt.xlabel('measured counts')
plt.ylabel('relative uncertainty')
plt.xlim([0,64e3])
plt.ylim([0,0.1])
plt.grid()
plt.legend()
plt.savefig('error_highres_IR',dpi=600)
plt.show()

# #%%
plot_order = np.array([12,14,15,13,16,17])
for i in plot_order:
    plt.plot(np.arange(0,lin_max_list[i]),non_lin_std[i,:lin_max_list[i]]/non_lin_fit[i,:lin_max_list[i]],color=cmap(items[i]['CCDSEL']),label='channel ' + str(items[i]['CCDSEL']))
    plt.plot(np.arange(lin_max_list[i]+1,len(non_lin_std[i,:])),non_lin_std[i,lin_max_list[i]:-1]/non_lin_fit[i,lin_max_list[i]:-1],color=cmap(items[i]['CCDSEL']),linestyle='dotted',label=None)
plt.title('Error for lowpixel macro')
plt.xlabel('measured counts')
plt.ylabel('relative uncertainty')
plt.xlim([0,64e3])
plt.ylim([0,0.1])
plt.grid()
plt.legend()
plt.savefig('error_lowpixel',dpi=600)
plt.show()

#%%
plot_order = np.array([18,20,21,19,22,23])
for i in plot_order:
    plt.plot(np.arange(0,lin_max_list[i]),non_lin_std[i,:lin_max_list[i]]/non_lin_fit[i,:lin_max_list[i]],color=cmap(items[i]['CCDSEL']),label='channel ' + str(items[i]['CCDSEL']))
    plt.plot(np.arange(lin_max_list[i]+1,len(non_lin_std[i,:])),non_lin_std[i,lin_max_list[i]:-1]/non_lin_fit[i,lin_max_list[i]:-1],color=cmap(items[i]['CCDSEL']),linestyle='dotted',label=None)
plt.title('Error for highres_UV macro')
plt.xlabel('measured counts')
plt.ylabel('relative uncertainty')
plt.xlim([0,64e3])
plt.ylim([0,0.1])
plt.legend()
plt.grid()
plt.savefig('error_highres_UV',dpi=600)
plt.show()

# %%
cmap = plt.get_cmap("tab10")
plt.plot(np.arange(8e3),np.arange(8e3),'k',label=None)
for i in range(0,6):
    plt.plot(np.arange(0,lin_max_list[i]),non_lin_fit[i,:lin_max_list[i]],color=cmap(items[i]['CCDSEL']),label='channel ' + str(items[i]['CCDSEL']))
    plt.plot(np.arange(lin_max_list[i]+1,len(non_lin_std[i,:])),non_lin_fit[i,lin_max_list[i]:-1],color=cmap(items[i]['CCDSEL']),linestyle='dotted',label=None)
plt.title('non_linearity for fullframe macro')
plt.xlabel('measured counts')
plt.ylabel('corrected counts')
plt.xlim([0,8e3])
plt.ylim([0,8e3])
plt.legend()
plt.grid()
plt.savefig('non_lin_fullframe',dpi=600)
plt.show()

cmap = plt.get_cmap("tab10")
plt.plot(np.arange(64e3),np.arange(64e3),'k',label=None)
plot_order = np.array([18,20,21,19,22,23])
for i in plot_order:
    plt.plot(np.arange(0,lin_max_list[i]),non_lin_fit[i,:lin_max_list[i]],color=cmap(items[i]['CCDSEL']),label='channel ' + str(items[i]['CCDSEL']))
    plt.plot(np.arange(lin_max_list[i]+1,len(non_lin_std[i,:])),non_lin_fit[i,lin_max_list[i]:-1],color=cmap(items[i]['CCDSEL']),linestyle='dotted',label=None)
plt.title('non_linearity for highres_UV macro')
plt.xlabel('measured counts')
plt.ylabel('corrected counts')
plt.xlim([0,64e3])
plt.ylim([0,64e3])
plt.legend()
plt.grid()
plt.savefig('non_lin_highres_UV',dpi=600)
plt.show()
