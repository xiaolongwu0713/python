from pre_all import *
import numpy as np
import mne
import os, re,sys,socket

data_dir = top_data_dir+'sEMG_zhangbin/'  # temp data dir
result_dir=data_dir+'training/'

ET_tmp=[3,4,8,9,12,17] # missing Data-L/TP008
#ET_tmp=[3,4,9,12]
PD_tmp=[5,7,10,13,14,15,16]
#PD_tmp=[5,7,10,13,14,15] # missing Data-R/TP016
others_tmp=[1,2,6,11,18,19,20]

ET=['TP'+"{0:03}".format(i) for i in ET_tmp]
PD=['TP'+"{0:03}".format(i) for i in PD_tmp]
others=['TP'+"{0:03}".format(i) for i in others_tmp]
NC=['TN'+"{0:03}".format(i+1) for i in range(18)]


# task labels
ET_task_labels={}
for ETi in ET:
    ET_task_labels[ETi]={}
    for handi in ['R','L']:
        ET_task_labels[ETi][handi]=list(range(15))
# task labels
PD_task_labels={}
for PDi in PD:
    PD_task_labels[PDi]={}
    for handi in ['R','L']:
        PD_task_labels[PDi][handi]=list(range(15))
# task labels
others_task_labels={}
for othersi in others:
    others_task_labels[othersi]={}
    for handi in ['R','L']:
        others_task_labels[othersi][handi]=list(range(15))
# task labels
NC_task_labels={}
for NCi in NC:
    NC_task_labels[NCi]={}
    for handi in ['R','L']:
        NC_task_labels[NCi][handi]=list(range(15))

#Left:
# TN004, missing Task 07 in Task 01 - 15.(6)
# TP008, missing Task 09 - 13 in Task 01 - 15.(8-12)
#
# Right:
# TP016, missing Task 08 - 13 in Task 01 - 15.(7-12)
NC_task_labels['TN004']['L'].pop(6)
tmp=ET_task_labels['TP008']['L']
ET_task_labels['TP008']['L']=tmp[:8]+tmp[13:]
tmp=PD_task_labels['TP016']['R']
PD_task_labels['TP016']['R']=tmp[:7]+tmp[13:]