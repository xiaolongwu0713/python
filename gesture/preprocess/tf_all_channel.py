import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
elif socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
from gesture.config import *

import hdf5storage
import os
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet
from mne.time_frequency import tfr_array_morlet

from gesture.config import *
from gesture.preprocess.chn_settings import get_channel_setting

if len(sys.argv)>1:
    sid = int(float(sys.argv[1]))

else: # debug in IDE
    sid=10

tf_dir=data_dir + '/tfAnalysis/P'+str(sid)+'/'
input_dir=data_dir+'preprocessing/P'+str(sid)+'/preprocessing2.mat'
if not os.path.exists(tf_dir):
    os.makedirs(tf_dir)

Session_num,UseChn,EmgChn,TrigChn = get_channel_setting(sid)
#original_fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == pn][0]
fs=1000
if sid==11 or sid==12:
    fs=500
mat=hdf5storage.loadmat(input_dir)
data = mat['Datacell']
good_channels=mat['good_channels']
channelNum=len(np.squeeze(good_channels))
data=np.concatenate((data[0,0],data[0,1]),0)
del mat

# stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
chn_names=np.append(["seeg"]*channelNum,["emg0","emg1","stim_trigger","stim_emg"])
chn_types=np.append(["seeg"]*channelNum,["emg","emg","stim","stim"])
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
raw = mne.io.RawArray(data.transpose(), info)

# gesture/events type: 1,2,3,4,5
events0 = mne.find_events(raw, stim_channel='stim_trigger')
events1 = mne.find_events(raw, stim_channel='stim_emg')
# events number should start from 0: 0,1,2,3,4, instead of 1,2,3,4,5
events0=events0-[0,0,1]
events1=events1-[0,0,1]

#print(events[:5])  # show the first 5
raw=raw.pick(["seeg"])

# 0 second means the gesture onset time, tmin=-4 means 4 seconds before onset.
# movement lasts for 5s from 0s onest.
epochs = mne.Epochs(raw, events1, tmin=-4, tmax=6,baseline=None)
# or epoch from 0s to 4s which only contain movement data.
# epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

# tf analysis on one epoch
epoch1=epochs['0'] # 20 trials. 8001 time points per trial for 8s.
#epoch2=epochs['1']
#epoch3=epochs['2']
#epoch4=epochs['3']
#epoch5=epochs['4']
#list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]
del epochs

#sub_ch_names=[epoch1.ch_names[i] for i in [1,2]]
sub_ch_names=epoch1.ch_names # uncomment this to analysis tf for all channels.

## frequency analysis
# define frequencies of interest (log-spaced)
fMin,fMax=2,150
fstep=1
freqs=np.arange(fMin,fMax,fstep) #148
fNum=freqs.shape[0]
#freqs = np.linspace(fMin,fMax, num=fNum)
cycleMin,cycleMax=8,50
cycleNum=fNum
#n_cycles = np.linspace(cycleMin,cycleMax, num=cycleNum)  # different number of cycle per frequency
n_cycles=freqs/2
#lowCycles=30
#n_cycles=[8]*lowCycles + [50]*(fNum-lowCycles)

averagePower=[] # access: averagePower[chn_index][mi/me]=2D, for example: averagePower[0][0]=2D tf data, 0th ch and 0th paradigm.
if fs==1000:
    decim=4
elif fs==500:
    decim=2
new_fs=fs/decim
for chIndex,chName in enumerate(sub_ch_names):
    if chIndex%20 == 0:
        print('TF analysis on '+str(chIndex)+'th channel.')
    # decim will decrease the sfreq, so 15s will becomes 5s afterward.
    averagePower.append(np.squeeze(tfr_morlet(epoch1, picks=[chIndex],
               freqs=freqs, n_cycles=n_cycles,use_fft=True,return_itc=False, average=True, decim=decim, n_jobs=1).data))
    #averagePower.append(tfr_morlet(epoch1, picks=[chIndex],
    #           freqs=freqs, n_cycles=n_cycles,use_fft=True,return_itc=False, average=True, decim=decim, n_jobs=1))

# exam the result
#fig, ax = plt.subplots()
#channel=0
#averagePower[channel].plot(baseline=(-3,-0.5), vmin=-4,vmax=4,mode='zscore', title=sub_ch_names[channel]+'_'+str(channel),axes=ax)

# crop the original power data because there is artifact at the beginning and end of the trial.
power=[] # power[0][0].shape: (148, 2000)
crop=0.5 #0.5s
shift=int(crop*new_fs)
crop1=0
crop2=15
for channel in range(len(sub_ch_names)):
    power.append([])
    power[channel]=averagePower[channel][:,shift:-shift]


time_to_onset=4 #s
baseline = [int(1),int(2.5*new_fs)]
tickpos=[int(i*new_fs) for i in [4-crop]]
ticklabel=['onset']

#(300, 5001)
vmin=-4
vmax=4
fig, ax = plt.subplots()
print('Ploting out to '+tf_dir+'.')
for channel in range(len(sub_ch_names)):
    if channel%20 == 0:
        print('Ploting '+str(channel)+'th channel.')
    base=power[channel][:,baseline[0]:baseline[1]] #base[0]:(148, 250)
    basemean=np.mean(base,1) #basemean[0]:(148,)
    basestd=np.std(base,1)
    power[channel]=(power[channel]-basemean[:,None])/basestd[:,None]

    im0=ax.imshow(power[channel],origin='lower',cmap='RdBu_r',vmin=vmin, vmax=vmax)
    ax.set_aspect('auto')

    ax.set_xticks(tickpos)
    ax.set_xticklabels(ticklabel)

    #plot vertical lines
    for x_value in tickpos:
        ax.axvline(x=x_value)
        ax.axvline(x=x_value)
    #fig.colorbar(im, ax=ax0)
    fig.colorbar(im0, orientation="vertical",fraction=0.046, pad=0.02,ax=ax)

    # save
    figname = tf_dir + 'tf_compare_'+str(channel) + '.png'
    fig.savefig(figname)

    # clean up plotting area
    ax.images[-1].colorbar.remove()
    ax.cla()
plt.close(fig)
filename=tf_dir+'tf_data'
np.save(filename,np.asarray(power))






