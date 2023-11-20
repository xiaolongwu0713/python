"""
Group electrodes by anatomical label
"""

import sys
import h5py
import hdf5storage
from mne.time_frequency import tfr_morlet
from sklearn import preprocessing
import numpy as np
import mne
import matplotlib.pyplot as plt
from gesture.config import *
from gesture.utils import read_sids

sids=read_sids()

#SignalChanel_Electrode_Registration.mat: Electrodes number, e.g.: 1,2,3,....110
#Electrode_Name_SEEG.mat: Electrode name, g.g.:  'POL B9'
#WholeCortex.mat: a lot of number, have no idea
#electrodes_Final.mat: List of code: A1, A2, A3...B1,B2...C1,C2...
ana_label_name={}
ana_label_name_unique_tmp=[]
for sid in sids:
    key=str(sid)
    filename=ele_dir+'P'+str(sid)+'/electrodes_Final_Norm.mat'
    mat=hdf5storage.loadmat(filename)
    mat=mat['elec_Info_Final_wm'][0]
    name=mat['name'][0,:]
    name=[i[0,0] for i in name]
    pos=mat['pos'][0,:]
    pos=[i[0,0] for i in pos]
    ano_pos=mat['ano_pos'][0,:]
    ano_pos=[i[0,0] for i in ano_pos]
    #ana_label_index=mat['ana_label_index'][0,:]
    #ana_label_index=[i[0,0] for i in ana_label_index]
    ana_label_name_tmp=mat['ana_label_name'][0,:]
    ana_label_name_tmp=[i[0,0] for i in ana_label_name_tmp]
    # two '--', replace with one '-'
    ana_label_name_tmp=[i.replace('--','-') for i in ana_label_name_tmp]
    ana_label_name[key]=ana_label_name_tmp
    ana_label_name_unique_tmp.extend(ana_label_name_tmp)
    #dis_to_ctx=mat['Dis_to_ctx'][0,:] # (190,)
    #norm_pos=mat['norm_pos'][0,:] # (190,)
    # no electrode in 'central' or 'frontal'
    if (not any(['central' in ai for ai in ana_label_name_tmp])) and (not any(['frontal' in ai for ai in ana_label_name_tmp])): # and 'frontal' in ai
        print("No central region: "+str(sid)+'.')

ana_unique=set(ana_label_name_unique_tmp)
ana_num=len(ana_unique)

groups={}
for ana in ana_unique:
    groups[ana]={}
    for sid in sids:
        key = str(sid)
        count=ana_label_name[key].count(ana)
        index=[i for i, val in enumerate(ana_label_name[key]) if val==ana]
        groups[ana][key] = index






