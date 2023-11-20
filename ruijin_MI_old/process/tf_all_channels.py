# tf_firstAvegThenZscore.py
import sys

from mne.time_frequency import tfr_morlet, tfr_multitaper
import numpy as np
import mne
import matplotlib.pyplot as plt
from ruijin_MI_old.config import *

# Epoch the data before doing this.

sid=2
data_dir='/Volumes/Samsung_T5/data/ruijin/MI/RJ_MI_Raw_Data/P2/tmp/'
plot_dir=data_dir + '/tfPlot/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

epochs=[]
epochs.append(mne.read_epochs(data_dir+'moveEpoch1.fif',preload=True).pick(picks=['seeg']).resample(1000))
epochs.append(mne.read_epochs(data_dir+'moveEpoch2.fif',preload=True).pick(picks=['seeg']).resample(1000))
ch_names=epochs[0].ch_names
sub_ch_names=[ch_names[i] for i in [108,109]]
sub_ch_names=ch_names # uncomment this to analysis tf for all channels.
singleMovementEpoch=[epoch.pick(picks=sub_ch_names) for epoch in epochs]

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
decim=4
new_fs=1000/decim
for chIndex,chName in enumerate(sub_ch_names):
    if chIndex%20 == 0:
        print('TF analysis on '+str(chIndex)+'th channel.')
    averagePower.append([])
    for i,paradigm in enumerate(['MI', 'ME']):
        # decim will decrease the sfreq, so 15s will becomes 5s afterward.
        averagePower[chIndex].append(np.squeeze(tfr_morlet(singleMovementEpoch[i], picks=[chIndex],
                   freqs=freqs, n_cycles=n_cycles,use_fft=True,return_itc=False, average=True, decim=decim, n_jobs=1).data))
# plot to test the cycle parameter
#fig, ax = plt.subplots()
#channel=0
#averagePower[channel].plot(baseline=(-1,0), vmin=-4,vmax=4,mode='zscore', title=ch_names[channel]+'_'+str(channel),axes=ax)

# crop the original power data because there is artifact at the beginning and end of the trial.
power=[] # power[0][0].shape: (148, 2000)
crop=0.5 #0.5s
shift=int(crop*new_fs)
crop1=0
crop2=15
for channel in range(len(sub_ch_names)):
    power.append([])
    for i,paradigm in enumerate(['MI', 'ME']):
        power[channel].append(averagePower[channel][i][:,shift:-shift])

'''
# use my own zscore function to plot because there is no different between mine and MNE.
# plot tf for all channels using MNE
for channel in range(len(ch_names)):
    fig, ax = plt.subplots()
    averagePower[channel].plot(baseline=(13,14.5), vmin=-4,vmax=4,mode='zscore', title=ch_names[channel]+'_'+str(channel),axes=ax)
    for x_value in movementLines:
        plt.axvline(x=x_value)
        plt.xticks(movementLines)
    figname=plot_dir+str(channel)+'.png'
    fig.savefig(figname, dpi=400)
    fig.clear()
    ax.clear()
    plt.close(fig)
'''

# return the index for the demanded freq
def getIndex(fMin,fMax,fstep,freq):
    freqs=[*range(fMin,fMax,fstep)]
    distance=[abs(fi-freq) for fi in freqs]
    index=distance.index(min(distance))
    return index

rest_duration=2.5 #s
baseline = [int((rest_duration-crop-1)*new_fs), int((rest_duration-crop)*new_fs)]
tickpos=[int(i*new_fs) for i in [0,2,3.5,4.5,8]]
ticklabel=['0.5','2.5','4','5','8.5']


#(300, 5001)
vmin=-4
vmax=4
fig, ax = plt.subplots(nrows=2,ncols=1, sharex=False)
#fig, ax = plt.subplots(2)
ax0=ax[0]
ax1=ax[1]
print('Ploting out to '+plot_dir+'.')
for channel in range(len(sub_ch_names)):
    if channel%20 == 0:
        print('Ploting '+str(channel)+'th channel.')
    base=[onep[:,baseline[0]:baseline[1]] for onep in power[channel]] #base[0]:(148, 250)
    basemean=[np.mean(_base,1) for _base in base] #basemean[0]:(148,)
    basestd=[np.std(_base,1) for _base in base]
    for i, paradigm in enumerate(['MI', 'ME']):
        power[channel][i]=power[channel][i]-basemean[i][:,None]
        power[channel][i]=power[channel][i]/basestd[i][:,None]
    im0=ax0.imshow(power[channel][0],origin='lower',cmap='RdBu_r',vmin=vmin, vmax=vmax)
    ax0.set_aspect('auto')
    im1=ax1.imshow(power[channel][1], origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax1.set_aspect('auto')

    ax0.set_xticks(tickpos)
    ax0.set_xticklabels(ticklabel)
    ax1.set_xticks(tickpos)
    ax1.set_xticklabels(ticklabel)

    #plot vertical lines
    for x_value in tickpos:
        ax0.axvline(x=x_value)
        ax1.axvline(x=x_value)
    #fig.colorbar(im, ax=ax0)
    fig.colorbar(im0, orientation="vertical",fraction=0.046, pad=0.02,ax=ax0)
    fig.colorbar(im1, orientation="vertical", fraction=0.046, pad=0.02, ax=ax1)

    # save
    figname = plot_dir + 'tf_compare_'+str(channel) + '.pdf'
    fig.savefig(figname)

    # clean up plotting area
    ax0.images[-1].colorbar.remove()
    ax1.images[-1].colorbar.remove()
    ax0.cla()
    ax1.cla()
plt.close(fig)


