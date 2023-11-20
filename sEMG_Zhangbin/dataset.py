import os
from pathlib import Path
import natsort
import numpy as np
import random,math
from sklearn.model_selection import train_test_split

ET_tmp = [3, 4, 8, 9, 12]  # missing Data-L/TP008
# ET_tmp=[3,4,9,12]
PD_tmp = [5, 7, 10, 13, 14, 15, 16]
PD_tmp = [5, 7, 10, 13, 14, 15]  # missing Data-R/TP016
others_tmp = [1, 2, 18, 19]

ET = ['TP' + "{0:03}".format(i) for i in ET_tmp]
PD = ['TP' + "{0:03}".format(i) for i in PD_tmp]
others = ['TP' + "{0:03}".format(i) for i in others_tmp]
NC = ['TN' + "{0:03}".format(i + 1) for i in range(18)]

channel_number = 7
hands_number = 2
task_number = 15
data_dir = 'H:/Long/data/sEMG_Zhangbin/'


# concatenate all subject
def windowing(data, wind_size,stride): #data shape: (48000, 7)
    windowed=[]
    s = 0
    total_len=data.shape[0]
    while stride * s + wind_size < total_len:
        start = s * stride
        tmp = data[start:(start + wind_size),:]
        windowed.append(tmp)
        s = s + 1
    # add the last window
    last_s = s - 1
    if stride * last_s + wind_size < total_len - 100: # discard the rest if too short data remaining
        tmp = data[-wind_size:,:]
        windowed.append(tmp)

    return np.asarray(windowed)



def get_data_trial_list():
    ET_data_raw = np.load(data_dir+'ET_data.npy',allow_pickle='TRUE').item()
    PD_data_raw = np.load(data_dir+'PD_data.npy',allow_pickle='TRUE').item()
    others_data_raw = np.load(data_dir+'others_data.npy',allow_pickle='TRUE').item()
    NC_data_raw = np.load(data_dir+'NC_data.npy',allow_pickle='TRUE').item()


    ET_data_tmp=[]
    PD_data_tmp=[]
    others_data_tmp=[]
    NC_data_tmp=[]

    # [ 145 * (48000,7)]
    for sub in ET_data_raw.keys():
        for handi in ['L','R']:
            for taski in ET_data_raw[sub][handi]:
                ET_data_tmp.append(taski)
    for sub in PD_data_raw.keys():
        for handi in ['L','R']:
            for taski in PD_data_raw[sub][handi]:
                PD_data_tmp.append(taski)

    for sub in others_data_raw.keys():
        for handi in ['L','R']:
            for taski in others_data_raw[sub][handi]:
                others_data_tmp.append(taski)

    for sub in NC_data_raw.keys():
        for handi in ['L','R']:
            for taski in NC_data_raw[sub][handi]:
                NC_data_tmp.append(taski)

    return ET_data_tmp,PD_data_tmp,others_data_tmp,NC_data_tmp


# train/val/test=6/2/2;
def five_CV(size,label):  #label is no use here
    #size=len(data)
    indexes=list(range(size))
    # 5-fold CV: shuffle --> fold
    random.shuffle(indexes)
    size_foldi=math.ceil(size/5)
    test_ind=[]
    val_ind=[]
    train_ind=[]
    for f in range(5):
        test_ind.append(indexes[f*size_foldi:(f+1)*size_foldi])
        rest_ind=indexes[0:f*size_foldi]+indexes[(f+1)*size_foldi:]
        val_ind.append(rest_ind[:size_foldi])
        train_ind.append(rest_ind[size_foldi:])
    assert sum(test_ind[0])+sum(val_ind[0])+sum(train_ind[0])==sum(test_ind[1])+sum(val_ind[1])+sum(train_ind[1])\
            ==sum(test_ind[2])+sum(val_ind[2])+sum(train_ind[2])==sum(test_ind[3])+sum(val_ind[3])+sum(train_ind[3])\
            ==sum(test_ind[4])+sum(val_ind[4])+sum(train_ind[4])
    return train_ind, val_ind,test_ind


# ET_data_tmp,PD_data_tmp,others_data_tmp,NC_data_tmp=get_data_trial_list()
def split_data(ET_data_tmp, PD_data_tmp,others_data_tmp,NC_data_tmp):
    train_ind_ET, val_ind_ET, test_ind_ET = five_CV(len(ET_data_tmp),999)
    train_ind_PD, val_ind_PD, test_ind_PD = five_CV(len(PD_data_tmp),999)
    train_ind_others, val_ind_others, test_ind_others = five_CV(len(others_data_tmp),999)
    train_ind_NC, val_ind_NC, test_ind_NC = five_CV(len(NC_data_tmp),999)

    folds_data=[]
    for f in range(5):
        train_ET, val_ET, test_ET = [ET_data_tmp[i] for i in train_ind_ET[f]], [ET_data_tmp[i] for i in val_ind_ET[f]], \
                                    [ET_data_tmp[i] for i in test_ind_ET[f]]
        train_PD, val_PD, test_PD = [PD_data_tmp[i] for i in train_ind_PD[f]], [PD_data_tmp[i] for i in val_ind_PD[f]], \
                                    [PD_data_tmp[i] for i in test_ind_PD[f]]
        train_others, val_others, test_others = [others_data_tmp[i] for i in train_ind_others[f]], [others_data_tmp[i] for i in val_ind_others[f]], \
                                                [others_data_tmp[i] for i in test_ind_others[f]]
        train_NC, val_NC, test_NC = [NC_data_tmp[i] for i in train_ind_NC[f]], [NC_data_tmp[i] for i in val_ind_NC[f]], \
                                    [NC_data_tmp[i] for i in test_ind_NC[f]]
        folds_data.append([train_ET,train_ind_ET,val_ET,val_ind_ET,test_ET,test_ind_ET,train_PD,train_ind_PD,val_PD,val_ind_PD,test_PD,test_ind_PD, \
          train_others,train_ind_others,val_others,val_ind_others,test_others,test_ind_others,train_NC,train_ind_NC,val_NC,val_ind_NC,test_NC,test_ind_NC])
    return folds_data


'''
# below will cause data leakage: split before windowing to prevent data leakage

for sub in ET_data_raw.keys():
    for handi in ['L','R']:
        ET_data_tmp.append([windowing(taski,wind_size,stride) for taski in ET_data_raw[sub][handi]])
        
for sub in PD_data_raw.keys():
    for handi in ['L','R']:
        PD_data_tmp.append([windowing(taski,wind_size,stride) for taski in PD_data_raw[sub][handi]])

for sub in others_data_raw.keys():
    for handi in ['L','R']:
        others_data_tmp.append([windowing(taski,wind_size,stride) for taski in others_data_raw[sub][handi]])

for sub in NC_data_raw.keys():
    for handi in ['L','R']:
        NC_data_tmp.append([windowing(taski,wind_size,stride) for taski in NC_data_raw[sub][handi]])

# flatten the inner list
def flatten_list(aList):
    tmp = [] # 145, not 150?=10*15: one has only 10 tasks instead of 15 tasks
    for lst in aList:
        tmp.extend(lst)
    return np.concatenate(tmp) # (55306, 500, 7)

ET_data2=flatten_list(ET_data_tmp2)

ET_data=flatten_list(ET_data_tmp) # (55306, 500, 7)
PD_data=flatten_list(PD_data_tmp) # (78687, 500, 7)
others_data=flatten_list(others_data_tmp) # (44947, 500, 7)
NC_data=flatten_list(NC_data_tmp) # (184078, 500, 7)

np.save(data_dir+'ET_data_3D',ET_data)
np.save(data_dir+'PD_data_3D',PD_data)
np.save(data_dir+'others_data_3D',others_data)
np.save(data_dir+'NC_data_3D',NC_data)
'''