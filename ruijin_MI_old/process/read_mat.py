# two problems:
# 1. how to read .bdf event file
# 2. how to read value of hdf5 reference dataset
# solution: output events into different format using eeglab.

import h5py
from mne.time_frequency import tfr_morlet
from sklearn import preprocessing
import numpy as np
import mne
import matplotlib.pyplot as plt
from ruijin_MI_old.config import *
#print(data_dir)

sid=2
#TODO: how to read .bdf event file
#raw_file=data_dir+'2_rb_201220/Visual_MI/RUIBING_2020_12_20_12_58_59_51f0cfc0-6829-4f7f-9c0b-56e851ba79bd/1/data.bdf'
#raw=mne.io.read_raw_bdf(raw_file) # last channe is emg, the rest are SEEG.
#events_file=data_dir+'2_rb_201220/Visual_MI/RUIBING_2020_12_20_12_58_59_51f0cfc0-6829-4f7f-9c0b-56e851ba79bd/1/evt.bdf'
#events=mne.io.read_raw_bdf(events_file)
filename='/Volumes/Samsung_T5/data/ruijin/MI/RJ_MI_Raw_Data/P'+str(sid)+'/H1.mat'
mat = h5py.File(filename, 'r')
data = mat['EEG']
raw=np.array(data['data'])

events=data['event'] #it's a group, list(events.items()) to show element dataset.
a=np.array(events['latency'][:])

h5py.h5r.dereference(mat['/EEG/event/latency'])
c=mat['/EEG/event/latency']


latency=events['latency'] # dataset
type=events['type']
type1=np.array(events['type'])

d=mat[type1]
np.array(d)