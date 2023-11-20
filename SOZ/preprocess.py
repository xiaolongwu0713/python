import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal

from common_plot import *
from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity

from SOZ.config import *
from dsp.common_dsp import identify_noisy_channel

#sid=0
#participant=participants[sid]
participant='HJY'
data_type='raw_data_more' # raw_data/raw_data_more
preprocess_dir='preprocess_more' if data_type=='raw_data_more' else 'preprocess'
soz_channels=soz_channel_all[participant] # ['B1-4', 'C1-5']

## read data
filename=data_dir+data_type+'/'+participant+'.edf'
raw_edf=mne.io.read_raw_edf(filename)
sf = int(raw_edf.info['sfreq']) # 2000
chn=len(raw_edf.ch_names) # 242

## Exclude the irrelevant channels by visual inspection;
#  Then update the irrelevant channels in the config fig
# left click the irrelevant channel, and retrieve them from raw_edf.info['bads'];
mne.viz.plot_raw(raw_edf,n_channels=10,scalings=dict(eeg=20e-5))
irrelevant_ch=raw_edf.info['bads']
## or load irrelevant channels from the config fig
irrelevant_ch=exclude_channels_all[participant]
ch_names=[item for item in raw_edf.ch_names if item not in irrelevant_ch]
assert len(raw_edf.ch_names)==len(ch_names)+len(irrelevant_ch)
# raw_edf.get_channel_types()
raw_edf.pick_channels(ch_names)
# or directly drop channel
# raw_notch.drop_channels(raw_notch.info['bads'])

####  exclude the noisy channel (50 Hz)
exam_len=5*60 # seconds
data=raw_edf.get_data(start=0, stop=exam_len*sf)
noisy_channels=identify_noisy_channel(data, 50, 300, 1000)
noisy_ch_name=[raw_edf.ch_names[i] for i in noisy_channels] # ['EEG F10-Ref']
# update the config file
noisy_ch_name=noisy_channels_all[participant]
ch_names=[item for item in raw_edf.ch_names if item not in noisy_ch_name]
raw_edf.pick_channels(ch_names)

#### Down sample data to 1K Hz
down_to_fs=1000
#start=0
#duration=1*60*60 * sf # 1 hours
# load_data can not load partial data;
raw_edf.load_data() # or do not load data, but much slower
# raw_edf.decimate(2) # only epoch class has decimate() function
raw_edf.resample(down_to_fs) # memory error on a 32GB laptop; works on a 64GB machine
filename=data_dir+preprocess_dir+'/'+participant+'/'+participant+'_down_sample.fif'
raw_edf.save(filename)

#### filter out line noise
raw_edf=mne.io.read_raw_fif(filename)
freqs = (50,100,150,200,250,300,350,400,450)
raw_edf.load_data()
raw_edf = raw_edf.notch_filter(freqs=freqs)
filename=data_dir+preprocess_dir+'/'+participant+'/'+participant+'_down_sample_notched.fif'
raw_edf.save(filename)
# visually inspect the notch result
#fig1 = raw_edf.plot_psd(fmax=1000, average=True)
#fig2 = raw_notch.plot_psd(fmax=1000, average=True)


#### Use numpy decimate ####
## ?? It's about 6GB for 1 hour data in the npy format.
# decimate using numpy
start=0
duration=1*60*60 * sf # 1 hours
crop_data=raw_edf.get_data(start=start,stop=start+duration)
deci_data=signal.decimate(crop_data,q=2,axis=0) # mem error on 32 or 64 GB RAM machine; 2000/4=500 Hz raw_data # (231, 28800072)-->
filename=data_dir+data_type+'/'+participant+'_decimated0.fif'
np.save(filename,deci_data)

start=duration
crop_data=raw_edf.get_data(start=start,stop=start+duration)
deci_data=signal.decimate(crop_data,q=2,axis=0) # mem error on 32 or 64 GB RAM machine; 2000/4=500 Hz raw_data # (231, 28800072)-->
filename=data_dir+data_type+'/'+participant+'_decimated1.fif'
np.save(filename,deci_data)

start=duration*2
crop_data=raw_edf.get_data(start=start,stop=start+duration)
deci_data=signal.decimate(crop_data,q=2,axis=0) # mem error on 32 or 64 GB RAM machine; 2000/4=500 Hz raw_data # (231, 28800072)-->
filename=data_dir+data_type+'/'+participant+'_decimated2.fif'
np.save(filename,deci_data)

start=duration*3
crop_data=raw_edf.get_data(start=start,stop=start+duration)
deci_data=signal.decimate(crop_data,q=2,axis=0) # mem error on 32 or 64 GB RAM machine; 2000/4=500 Hz raw_data # (231, 28800072)-->
filename=data_dir+data_type+'/'+participant+'_decimated3.fif'
np.save(filename,deci_data)


soz_channels = soz_channel_all[participant]
soz_indexes = []  # [12, 13, 14, 15, 28, 29, 30, 31, 32]
for i in range(len(soz_channels)):
    search=np.where(np.asarray(raw_edf.info['ch_names']) == soz_channels[i])[0]
    if len(search) >0: # maybe the SOZ channel is excluded because of the line noise
        soz_indexes.append(search[0])
non_SOZ_index = np.setdiff1d(np.arange(len(raw_edf.info['ch_names'])), soz_indexes).tolist()  # memory full
labels={'soz_index':soz_indexes,
       'non_soz_index':non_SOZ_index}
filename = data_dir  + 'label/' + participant + '/'+participant+'_labels.npy'
np.save(filename,labels)




