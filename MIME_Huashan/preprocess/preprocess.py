import re

import mne
import matplotlib.pyplot as plt
import numpy as np
from MIME_Huashan.config import data_dir

sub_names=['HHFU016_0714','017_0725','0815_Ma_JinLiang_0815','HHFU22_0902','HHFU026_1102','HHFU027_motorimagery']
# 'HHFU027_motorimagery': a,b,c,d,e,f,g,h shafts; each have 8 contacts; 8*8=64 electrodes;
sub_name=sub_names[-1]
filename=data_dir+'raw/'+sub_name+'/eeg.edf'
raw = mne.io.read_raw_edf(filename,preload=True)
#ch_index_str=[str(chi)+'_'+channels[chi].strip() for chi in [*range(len(channels))]]
#ch_index=[*range(len(channels))] #147

'''
channels are grouped into bad/EMG/DC/EEG;
'''
if sub_names=='HHFU016':
    ch_names = raw.ch_names
    ch_types=[]
    for c in ch_names:
        if 'POL DC' in c:
            ch_types.append('stim')
        elif 'EKG' in c or 'EMG' in c:
            ch_types.append('emg')
        else:
            ch_types.append('eeg')
elif sub_names=='017':
    raw.plot() # pick the bad channels
    # 18 bad chanenls
    bad_channels=['POL E','POL SPZ','POL EKG2','POL SCZ','POL SFZ','POL SF4','POL SC4','POL SP4','EEG SO2-Ref','POL EKG1','POL EMGL1','POL EMGL2',
    'POL EMGR1','POL EMGR2','POL BP1','POL BP2','POL BP3','POL BP4']
    bad_channels = raw.info['bads']
    #raw.pick(picks='all', exclude='bads') # or
    raw.drop_channels(bad_channels) # 98-->80

    raw.plot() # click the emg channels
    emg_channels=raw.info['bads'] # 8
    emg_channels= ['POL E7-1', 'POL E8-1', 'POL E9', 'POL E10', 'POL E12', 'POL E11', 'POL E13', 'POL E14']

    ch_names=raw.ch_names
    ch_types = []
    for c in ch_names:
        if 'POL DC' in c:
            ch_types.append('stim')
        elif c in emg_channels:
            ch_types.append('emg')
        else:
            ch_types.append('eeg')

elif sub_names=='HHFU027_motorimagery': # channel names contain string [a,b,c,d,e,f,g,h][1,2,3,4,5,6,7,8];
    bad_channels=['POL E','POL EKG1','POL EKG2','POL EMGL1','POL EMGL2','POL EMGR1','POL EMGR2',
                  'POL SS1','POL SS2','POL SS3','POL SS4','POL SS5','POL SS6','POL SS7','POL SS8','POL SS9','POL SS10',
                  'POL BP1','POL BP2','POL BP3','POL BP4']
    emg_channels=['POL B27','POL B28','POL B29','POL B30','POL B31','POL B32','POL B33','POL B34']
    use_channels=[]
    DC_chanenls=[]
    ch_types=[]

    for chi in raw.ch_names:
        if 'POL DC' in chi:
            ch_types.append('stim')
        elif chi in emg_channels:
            ch_types.append('emg')
        else:
            ch_types.append('eeg')
        #elif re.match(".*[A-G][1-8].*",chi) and not any([i in chi for i in emg_channels+['EKG','SS','BP']]):



plot_channels=10
raw.plot(duration=100,time_format='clock',n_channels=plot_channels, scalings=dict(eeg=50e-5))

fs = round(raw.info['sfreq'])
info = mne.create_info(ch_names=raw.ch_names, ch_types=raw.get_channel_types(), sfreq=fs)
raw = mne.io.RawArray(raw.get_data(), info)
raw.resample(sfreq=1000)

# EEG
raw_eeg=raw.copy().pick(picks=['eeg'])
filename=data_dir+'preprocess/'+sub_name+'/raw_eeg.fif'
raw_eeg.save(filename,overwrite=True)

# load again and do some signal process on the EEG channel
filename=data_dir+'preprocess/'+sub_name+'/raw_eeg.fif'
raw_eeg = mne.io.read_raw_fif(filename,preload=True)
raw_eeg.plot()
raw_eeg.notch_filter(np.arange(50, 451, 50),picks=['all'])
raw_eeg.plot()
if sub_names=='HHFU016':
    bad_channels=[] # 5
    for c in raw_eeg.ch_names: # 143
        if 'BP' in c:
            bad_channels.append(c)
    bad_channels.append('19_POL E')
    good_channels=[] # 138
    for c in raw_eeg.ch_names:
        if c not in bad_channels: # 141
            good_channels.append(c)
    raw_eeg.info['bads']=bad_channels

    raw_eeg.pick(picks=good_channels)
    raw_eeg.plot()
filename=data_dir+'preprocess/'+sub_name+'/raw_eeg_processed.fif'
raw_eeg.save(filename,overwrite=True)

# EMG
raw_emg=raw.copy().pick(picks=['emg'])
filename=data_dir+'preprocess/'+sub_name+'/raw_emg.fif'
raw_emg.save(filename,overwrite=True)
# DC
raw_dc=raw.copy().pick(picks=['stim'])
filename=data_dir+'preprocess/'+sub_name+'/raw_dc.fif'
raw_dc.save(filename,overwrite=True)

## analysis the DC channels
filename=data_dir+'preprocess/'+sub_name+'/raw_dc.fif'
raw_dc = mne.io.read_raw_fif(filename,preload=True)
fs=1000
#dc_raw=raw.copy().pick_types(stim=True)
# MNE only perform plot_psd on 'data' channel, not the stim channels. Have to convert the 'stim' type to 'eeg' type
info_tmp = mne.create_info(ch_names=raw_dc.ch_names, ch_types=['eeg']*len(raw_dc.ch_names), sfreq=fs)
raw_dc = mne.io.RawArray(raw_dc.get_data(), info_tmp)
raw_dc=raw_dc.notch_filter(np.arange(50, 251, 50),picks=['all'])
#raw_dc.plot_psd(tmin=25, tmax=30,picks=['all']) # still some residual line noise at 50hz,100hz,150hz and 200hz
raw_dc.plot()
## sum over all DC channels
info_tmp = mne.create_info(ch_names=['dc_sum'], ch_types=['stim'], sfreq=fs)
dc_sum = mne.io.RawArray(raw_dc.get_data().sum(axis=0)[np.newaxis,:], info_tmp)
dc_sum.plot()
#events=mne.find_events(dc_sum,stim_channel='dc_sum',min_duration=0.001) # missed some triggers
trigger_signal=dc_sum.get_data().squeeze() # (1, 3854105)
filename=data_dir+'preprocess/'+sub_name+'/trigger_signal_sum.npy'
np.save(filename,trigger_signal)

# load data directly
filename=data_dir+'preprocess/'+sub_name+'/trigger_signal_sum.npy'
trigger_signal=np.load(filename).squeeze() # (3854105,)
fs=1000
climb=10 # minimum 5
hight=5
#width=
if sub_names=='HHFU016':
    start=int(220*fs) # samples
    end=int(1800*fs)
elif sub_names=='HHFU016':
    start = 0
    end = len(trigger_signal)-10
trigger=[]
i=start
while i < end:
    if trigger_signal[i]<1 and trigger_signal[i+climb]>2:
        trigger.append(i)
        i=i+climb
    i=i+1

# visually verify the trigger
fig,ax=plt.subplots()
ax.clear()
ax.plot(trigger_signal)
for i in range(len(trigger)):
    ax.axvline(x=trigger[i], color='red', linestyle='--')

# Subject:HHFU016: remove the experiment start marker
if sub_names=='HHFU016':
    session_marker=[0,33,66,99]
    tmp=[]
    for i in range(len(trigger)):
        if i not in session_marker:
            tmp.append(trigger[i])
    trigger=np.asarray(tmp)

filename=data_dir+'preprocess/'+sub_name+'/trigger.npy'
np.save(filename,trigger)

# read markers
import scipy.io
if sub_name=='HHFU016':
    sessions=4
elif sub_name=='017':
    sessions = 5
markers_tmp=[]
for s in range(sessions):
    filename=data_dir+'raw/'+sub_name+'/trigger/session'+str(s+1)+'.mat'
    mat=scipy.io.loadmat(filename)['marker']
    markers_tmp.append(list(mat[0,:]))
markers=[item for sublist in markers_tmp for item in sublist] # 128

events=np.ones((len(markers),3),dtype=int)
for i in range(len(trigger)):
    events[i,:]=[trigger[i],0,markers[i]]
filename=data_dir+'preprocess/'+sub_name+'/events.eve'
mne.write_events(filename, events,overwrite=True)


# visually check the alignment between the extracted trigger and DC channels
filename=data_dir+'preprocess/'+sub_name+'/raw_dc.fif'
raw_dc = mne.io.read_raw_fif(filename,preload=True)
filename=data_dir+'preprocess/'+sub_name+'/events.eve'
events=mne.read_events(filename)
raw_dc.plot(
    events=events,
    start=5,
    duration=10,
    color="gray",
    event_color={1: "r", 2: "g", 3: "b", 4: "m", 5: "y", 6: "k", 7: "r", 8: "g"},
)





