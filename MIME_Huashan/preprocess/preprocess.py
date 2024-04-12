import re
import scipy.io
import mne
import matplotlib.pyplot as plt
import numpy as np
from MIME_Huashan.config import data_dir

sub_names=['1_HHFU016_0714','2_017_0725','3_0815_Ma_JinLiang_0815','4_HHFU22_0902','5_HHFU026_1102','6_HHFU027_motorimagery']
# 'HHFU027_motorimagery': a,b,c,d,e,f,g,h shafts; each have 8 contacts; 8*8=64 electrodes;
sid=6
inspect=False
sub_name=sub_names[sid-1]
if sid==5:
    file1 = data_dir + 'raw/' + sub_name + '/session1.edf'
    file2 = data_dir + 'raw/' + sub_name + '/session2.edf'
    file3 = data_dir + 'raw/' + sub_name + '/session3.edf'
    file4 = data_dir + 'raw/' + sub_name + '/session4.edf'
    raw1 = mne.io.read_raw_edf(file1)
    raw2 = mne.io.read_raw_edf(file2)
    raw3 = mne.io.read_raw_edf(file3)
    raw4 = mne.io.read_raw_edf(file4)
    raw=mne.concatenate_raws([raw1,raw2,raw3,raw4], preload=None, on_mismatch='raise', verbose=None)

else:
    filename=data_dir+'raw/'+sub_name+'/eeg.edf'
    raw = mne.io.read_raw_edf(filename,preload=True)
sf_tmp = round(raw.info['sfreq'])
sf=1000
raw.resample(sfreq=sf)
#ch_index_str=[str(chi)+'_'+channels[chi].strip() for chi in [*range(len(channels))]]
#ch_index=[*range(len(channels))] #147

'''
channels are grouped into bad/EMG/DC/EEG;
'''
# if sid==1:
#     ch_names = raw.ch_names
#     ch_types=[]
#     for c in ch_names:
#         if 'POL DC' in c:
#             ch_types.append('stim')
#         elif 'EMG' in c:
#             ch_types.append('emg')
#         elif 'POL BP' in c or 'EKG' in c or 'POL E' in c: # 'POL E' stright line
#             ch_types.append('misc')
#         else:
#             ch_types.append('eeg')
if sid==1:
    raw.plot() # click channel name to the bad channels, close the plot to save
    bad_channels = raw.info['bads']
    bad_channels = ['POL E', 'POL EKG1', 'POL EKG2', 'POL BP1', 'POL BP2', 'POL BP3', 'POL BP4']
    raw.drop_channels(bad_channels) # 165-158

    raw.plot()  # click to pick the emg channels
    emg_channels = raw.info['bads']  # 8
    emg_channels = ['POL EMGL1', 'POL EMGL2', 'POL EMGR1', 'POL EMGR2']

    ch_names = raw.ch_names
    ch_types = []
    for c in ch_names:
        if 'POL DC' in c:
            ch_types.append('stim')
        elif c in emg_channels:
            ch_types.append('emg')
        else:
            ch_types.append('eeg')

elif sid==2:
    raw.plot() # pick the bad channels
    # 18 bad chanenls
    bad_channels = raw.info['bads']
    bad_channels=['POL E','POL SPZ','POL EKG2','POL SCZ','POL SFZ','POL SF4','POL SC4','POL SP4','EEG SO2-Ref','POL EKG1','POL EMGL1','POL EMGL2',
    'POL EMGR1','POL EMGR2','POL BP1','POL BP2','POL BP3','POL BP4']

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

elif sid==3: # total 98 channels
    raw.plot() # pick the bad channels
    # 18 bad chanenls
    bad_channels = raw.info['bads']
    bad_channels=['POL E','POL SFZ','POL SCZ','POL SPZ','POL SF3','POL SC3','POL SP3','EEG SO1-Ref','POL EKG1','POL EKG2','POL EMGL1',
     'POL EMGL2','POL EMGR1','POL EMGR2','POL BP1','POL BP2','POL BP3','POL BP4']

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
elif sid==4: # total 98 channels
    raw.plot() # pick the bad channels
    raw.notch_filter(np.arange(50, 251, 50), picks=['all']) # strong line noice
    raw_highpass = raw.copy().filter(l_freq=6, h_freq=None)
    raw_highpass.plot()
    raw.compute_psd().plot()
    # 18 bad chanenls
    bad_channels = raw.info['bads']
    bad_channels=['POL EKG1','POL EKG2','POL EMGL1','POL EMGL2','POL EMGR1','POL EMGR2','EEG FZ-Ref','EEG CZ-Ref','EEG PZ-Ref']

    #raw.pick(picks='all', exclude='bads') # or
    raw.drop_channels(bad_channels) # 98-->80

    raw.plot() # click the emg channels
    emg_channels=raw.info['bads'] # 8
    emg_channels= ['POL E10','POL E11','POL E12','POL E13','POL E14','POL E15','POL E16','EEG F1-Ref-1']

    ch_names=raw.ch_names
    ch_types = []
    for c in ch_names:
        if 'POL DC' in c:
            ch_types.append('stim')
        elif c in emg_channels:
            ch_types.append('emg')
        else:
            ch_types.append('eeg')

elif sid==5: # total 98 channels
    raw.plot() # pick the bad channels

    # 18 bad chanenls
    bad_channels = raw.info['bads']
    bad_channels=['POL E','POL EKG1','POL EKG2','POL EMGL1','POL EMGL2','POL EMGR1','POL EMGR2','POL BP1','POL BP2','POL BP3','POL BP4']
    bad_channels.append('POL B8') # psd show this channel is corrupted
    #raw.pick(picks='all', exclude='bads') # or
    raw.drop_channels(bad_channels) # 115-11=104

    raw.plot() # click the emg channels
    emg_channels=raw.info['bads'] # 8
    emg_channels= ['EEG F7-Ref-1','EEG F8-Ref-1','EEG F9-Ref','EEG F10-Ref','POL F11','POL F12','POL G1-1','POL G2-1']

    ch_names=raw.ch_names
    ch_types = []
    for c in ch_names:
        if 'POL DC' in c:
            ch_types.append('stim')
        elif c in emg_channels:
            ch_types.append('emg')
        else:
            ch_types.append('eeg')

elif sid==6: # channel names contain string [a,b,c,d,e,f,g,h][1,2,3,4,5,6,7,8];
    raw.plot()  # pick the bad channels
    bad_channels = raw.info['bads']
    bad_channels=['POL E','POL EKG1','POL EKG2','POL EMGL1','POL EMGL2','POL EMGR1','POL EMGR2',
                  'POL SS1','POL SS2','POL SS3','POL SS4','POL SS5','POL SS6','POL SS7','POL SS8','POL SS9','POL SS10',
                  'POL BP1','POL BP2','POL BP3','POL BP4']
    # raw.pick(picks='all', exclude='bads') # or
    raw.drop_channels(bad_channels)  # 109-21=88

    raw.plot()  # click the emg channels
    emg_channels = raw.info['bads']  # 8
    emg_channels = ['POL B27','POL B28','POL B29','POL B30','POL B31','POL B32','POL B33','POL B34']

    ch_names = raw.ch_names
    ch_types = []
    for c in ch_names:
        if 'POL DC' in c:
            ch_types.append('stim')
        elif c in emg_channels:
            ch_types.append('emg')
        else:
            ch_types.append('eeg')
info = mne.create_info(ch_names=raw.ch_names, ch_types=ch_types, sfreq=sf)
raw = mne.io.RawArray(raw.get_data(), info)
stim_ch_num=sum([i=='stim' for i in ch_types])  # 16
emg_ch_num=sum([i=='emg' for i in ch_types]) # 4/8

plot_channels=10
raw.plot(duration=100,time_format='clock',n_channels=plot_channels, scalings=dict(eeg=50e-5))

# EEG
raw_eeg=raw.copy().pick(picks=['eeg'])
filename=data_dir+'preprocess/'+sub_name+'/raw_eeg.fif'
raw_eeg.save(filename,overwrite=True)

# load again and do some signal process on the EEG channel
#filename=data_dir+'preprocess/'+sub_name+'/raw_eeg.fif'
#raw_eeg = mne.io.read_raw_fif(filename,preload=True)

raw_eeg.notch_filter(np.arange(50, 451, 50),picks=['all'])
raw_eeg.compute_psd().plot() # check again for any corrupted channels, and drop them.
#raw_eeg.drop_channels(['POL B8'])
filename=data_dir+'preprocess/'+sub_name+'/raw_eeg_notched.fif'
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
#raw_dc = mne.io.read_raw_fif(filename,preload=True)

# MNE only perform plot_psd on 'data' channel, not the stim channels. Have to convert the 'stim' type to 'eeg' type
# maybe don't need to notch the stim, just sum them up
info_tmp = mne.create_info(ch_names=raw_dc.ch_names, ch_types=['eeg']*len(raw_dc.ch_names), sfreq=sf)
raw_dc = mne.io.RawArray(raw_dc.get_data(), info_tmp)
raw_dc=raw_dc.notch_filter(np.arange(50, 251, 50),picks=['all'])
#raw_dc.plot_psd(tmin=25, tmax=30,picks=['all']) # still some residual line noise at 50hz,100hz,150hz and 200hz
raw_dc.plot()

## sum over all DC channels
info_tmp = mne.create_info(ch_names=['dc_sum'], ch_types=['stim'], sfreq=sf)
dc_sum = mne.io.RawArray(raw_dc.get_data().sum(axis=0)[np.newaxis,:], info_tmp)
dc_sum.plot(time_format='float')
#events=mne.find_events(dc_sum,stim_channel='dc_sum',min_duration=0.001) # missed some triggers
trigger_signal=dc_sum.get_data().squeeze() # (1, 3854105)
filename=data_dir+'preprocess/'+sub_name+'/trigger_signal_sum.npy'
np.save(filename,trigger_signal)

# load data directly
filename=data_dir+'preprocess/'+sub_name+'/trigger_signal_sum.npy'
sf=1000

climb=10 # minimum 5
if sid==1:
    start=int(220*sf) # samples
    end=int(1800*sf)
elif sid==2 or sid==4 or sid==6:
    start = 0
    end = len(trigger_signal)-10
elif sid==3:
    start=100000
    end=len(trigger_signal)-10
elif sid==5:
    start=10000
    end = len(trigger_signal) - 10
trigger=[]
i=start
while i < end:
    if trigger_signal[i]<1 and trigger_signal[i+climb]>2:
        trigger.append(i)
        i=i+2*sf # shouldn't be any trigger after one second of current one
    i=i+1

## should be 128=4sessions * 8tasks * 4repetition
## or 120=5sessions * 8tasks * 3repetitions
trial_number=len(trigger) #132;
# Subject:HHFU016: remove the experiment start marker
if sid==1 or sid==3 or sid==4:
    session_marker=[0,33,66,99]
    tmp=[]
    for i in range(len(trigger)):
        if i not in session_marker:
            tmp.append(trigger[i])
    trigger=np.asarray(tmp)
elif sid==5: # 3sessions*6repeitition*8 + 1session*8repetition*8=3*48 + 64=144+64=208
    session_marker = [0,49,98]
    tmp = []
    for i in range(len(trigger)):
        if i not in session_marker:
            tmp.append(trigger[i])
    trigger = np.asarray(tmp)
elif sid==6: # 3sessions*6repeitition*8 + 1session*8repetition*8=3*48 + 64=144+64=208
    session_marker = [0,49,98,146]
    tmp = []
    for i in range(len(trigger)):
        if i not in session_marker:
            tmp.append(trigger[i])
    trigger = np.asarray(tmp)

# visually verify the trigger
fig,ax=plt.subplots()
ax.clear()
ax.plot(trigger_signal)
for i in range(len(trigger)):
    ax.axvline(x=trigger[i], color='red', linestyle='--')

figname=data_dir+'preprocess/'+sub_name+'/trigger.pdf'
fig.savefig(figname)
filename=data_dir+'preprocess/'+sub_name+'/trigger.npy'
np.save(filename,trigger)

# read markers
if sid==1 or sid==3 or sid==4 or sid==5 or sid==6:
    sessions=4
    marker_name='marker_shown'
elif sid==2:
    sessions = 5
    marker_name='marker'

markers_tmp=[]
for s in range(sessions):
    filename=data_dir+'raw/'+sub_name+'/trigger/session'+str(s+1)+'.mat'
    mat=scipy.io.loadmat(filename)[marker_name]
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



