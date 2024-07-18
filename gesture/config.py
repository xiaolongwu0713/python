from pre_all import *
import numpy as np
import mne
import os, re,sys,socket

data_dir = top_data_dir+'gesture/'  # temp data dir
root_dir = top_root_dir+'gesture/'
result_dir = root_dir + 'result/'
meta_dir = top_meta_dir+'gesture/'
ele_dir = meta_dir + 'EleCTX_Files/'
info_dir = meta_dir + 'info/'
tmp_dir=tmp_dir+'gesture/'

default_frequency=1000
total_sub_trail_after_windowing=380 # 380 * 5 = 1900
# participants details
classNum = 5
class_number=5
good_sids=[2,3,4,10,13,17,18,29,32,41]
good_sids_file=meta_dir+'good_sids.txt'
with open(good_sids_file, "r") as infile:
    lines = infile.read().split('\n')
good_sids=[int(tmpi) for tmpi in lines if len(tmpi)>0] # [2, 3, 4, 10, 13, 17, 18, 25, 29, 32, 34, 41]

final_good_sids_file=meta_dir+'final_good_sids.txt'
with open(final_good_sids_file, "r") as infile:
    lines = infile.read().split('\n')
final_good_sids=[int(tmpi) for tmpi in lines if len(tmpi)>0]

Frequencies = np.load(info_dir+'Info(use_Info.txt_instead).npy', allow_pickle=True)

# [sid, fs, number of electrodes]
Electrodes_tmp=np.array([[2, 1000,121], [3, 1000,180], [ 4, 1000,60], [5, 1000,178], [ 7, 1000,143],[ 8, 1000,169],
    [ 9, 1000,114], [ 10, 2000,208],[11,500,194],[12,500,94],[13, 2000,102], [14, 2000,130],[ 16, 2000,170],
    [ 17, 2000,144], [ 18, 2000,144], [ 19, 2000,137], [ 20, 1000,108], [ 21, 1000,118], [ 22, 2000,150],
    [ 23, 2000,198],[24, 2000,130],[ 25, 2000,137], [ 26, 2000,154],[  29, 2000,110], [ 30, 2000,108], [ 31, 2000,72],
    [ 32, 2000,56], [ 34, 2000,102], [ 35, 1000,136],[36, 2000,117],[ 37, 2000,64],[38,2000,242],[39, 2000,126], [41, 2000,190]])
Electrodes={}
for e in Electrodes_tmp:
    sid=e[0]
    Electrodes[sid]=e[2]

active_channels={}
active_channels['10']=range(162,168,1)
active_channels['41']=[*range(25,38)]+[77,96,97]+[*range(156,170)]

fbands=[] #delta, theta, alpha,beta,gamma
fbands.append([0.5, 4])
fbands.append([4, 8])
fbands.append([8, 12]) # mu(motor cortex)/alpha(visual cortex)
fbands.append([13, 30]) # beta
fbands.append([60, 125]) # genBandPower_znormalied.py
#fbands.append([60, 140]) # gamma: 55-85

ERD=[13,30]
ERS=[55,100]

def printVariables(variable_names):
    for k in variable_names:
        max_name_len = max([len(k) for k in variable_names])
        print(f'  {k:<{max_name_len}}:  {globals()[k]}')



