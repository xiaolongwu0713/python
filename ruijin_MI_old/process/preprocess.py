'''
Process workflow:
'''

import h5py
import hdf5storage
from mne.time_frequency import tfr_morlet
from sklearn import preprocessing
import numpy as np
import mne
import matplotlib.pyplot as plt
from ruijin_MI_old.config import *
#print(data_dir)

sid=2
fs=2000
filename='/Volumes/Samsung_T5/data/ruijin/MI/Raw_data/P'+str(sid)+'/tmp/v_data.mat'
mat = h5py.File(filename, 'r')
data=np.array(mat['data']).transpose()# (3570000, 180), 179*seeg + 1 EMG

### create info and raw data
ch_names=np.append(['seeg']*179,['emg']) # events, emg. total 112 channels = 110+2
#ch_types=np.repeat(np.array('seeg'),126)
ch_types=ch_names
info = mne.create_info(ch_names=list(ch_names), ch_types=list(ch_types), sfreq=fs)
raw = mne.io.RawArray(data, info)
emg=data[-1,:]

# notch filter
freqs = (50,150,250,350)
raw.notch_filter(freqs=freqs) # notch seeg and emg
#raw.plot_psd(tmin=None, tmax=None,picks=['seeg'],average=True,spatial_colors=False,color='red')


filename='/Volumes/Samsung_T5/data/ruijin/MI/Raw_data/P'+str(sid)+'/tmp/v_event_type.mat'
type_mat=hdf5storage.loadmat(filename)['event_type'][0]
event_type=[]
for i in range(type_mat.shape[0]):
    tmp=type_mat[i][0][0]
    #pattern="[['(.*?)']]"
    #substring = re.search(pattern, tmp).group(1)
    event_type.append(int(float(tmp[11:])))

filename='/Volumes/Samsung_T5/data/ruijin/MI/Raw_data/P'+str(sid)+'/tmp/v_event_latency.mat'
latency_mat=hdf5storage.loadmat(filename)['event_latency'][0]
event_latency=[]
for i in range(latency_mat.shape[0]):
    tmp=latency_mat[i]
    #pattern="[['(.*?)']]"
    #substring = re.search(pattern, tmp).group(1)
    event_latency.append(int(float(tmp)))

# remove bad trials by comparring EMG signal with events type list
emg=raw.pick(picks=['emg']) # no emg



duration=[0]*len(event_latency)
events=np.asarray([event_latency,duration, event_type]).transpose()
# keep marker 21-26 only
sub_events=[]
for i in events:
    if i[2] in [21,22,23,24,25,26]:
        sub_events.append(list(i))

#epochs = mne.Epochs(raw, events, baseline=None)
epochs = mne.Epochs(raw, sub_events, tmin=-2.5, tmax=6.5,baseline=None)
epochs=epochs.resample(1000)
epoch1=epochs['21']
epoch2=epochs['22']
epoch3=epochs['23']
epoch4=epochs['24']
epoch5=epochs['25']
epoch6=epochs['26']

epoch1.save('/Volumes/Samsung_T5/data/ruijin/MI/preprocessing_data/P'+str(sid)+'/preprocessing/'+'moveEpoch1.fif', overwrite=True)
epoch2.save('/Volumes/Samsung_T5/data/ruijin/MI/preprocessing_data/P'+str(sid)+'/preprocessing/'+'moveEpoch2.fif', overwrite=True)
epoch3.save('/Volumes/Samsung_T5/data/ruijin/MI/preprocessing_data/P'+str(sid)+'/preprocessing/'+'moveEpoch3.fif', overwrite=True)
epoch4.save('/Volumes/Samsung_T5/data/ruijin/MI/preprocessing_data/P'+str(sid)+'/preprocessing/'+'moveEpoch4.fif', overwrite=True)
epoch5.save('/Volumes/Samsung_T5/data/ruijin/MI/preprocessing_data/P'+str(sid)+'/preprocessing/'+'moveEpoch5.fif', overwrite=True)
epoch6.save('/Volumes/Samsung_T5/data/ruijin/MI/preprocessing_data/P'+str(sid)+'/preprocessing/'+'moveEpoch6.fif', overwrite=True)
