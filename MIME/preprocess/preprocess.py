from mne.time_frequency import tfr_morlet
from sklearn import preprocessing

from grasp.process.utils import get_trigger, genSubTargetForce, getRawData, getMovement, getForceData, \
    get_trigger_normal, getRawDataInEdf, getMovement_sid10And16
import numpy as np
import mne
import matplotlib.pyplot as plt
from grasp.config import *
from grasp.process.channel_settings import *


filename='H:/Long/data/MIME/DC0201EQ.edf'
data = mne.io.read_raw_edf(filename,preload=True)
channels = data.ch_names
ch_index_str=[str(chi)+'_'+channels[chi].strip() for chi in [*range(len(channels))]]
ch_index=[*range(len(channels))] #147
ch_types=[]
for c in channels:
    if 'POL DC' in c:
        ch_types.append('stim')
    else:
        ch_types.append('seeg')

fs = round(data.info['sfreq'])
raw = data.get_data()
info = mne.create_info(ch_names=list(ch_index_str), ch_types=list(ch_types), sfreq=fs)
raw = mne.io.RawArray(raw, info)

raw.plot(scalings=dict(stim=1,seeg=1e-1),n_channels=5,duration=30.0,start=0)

dc_raw=raw.copy().pick_types(stim=True)
dc_raw.plot(scalings=dict(all=10000),duration=30,start=0)

dc_data=dc_raw.get_data() # 16 channels
dc1=dc_data[0,:]

fig,ax=plt.subplots()
ax.plot(dc_data.transpose())
ax.plot(dc_data[1,:])
ax.clear()
xx=np.arange(0,dc_data.shape[1])/2000
ax.set_xticks(np.arange(0,dc_data.shape[1]))
ax.set_xticklabels(xx)
