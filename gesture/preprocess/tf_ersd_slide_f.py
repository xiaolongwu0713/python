'''
plot TF on top and ERSD on bottom.
Use tfr_morlet(average=True) to plot the TF;
Use tfr_morlet(average=False) to plot the ERS/D mean+-std across all trials.
Choose frequency window ers_wind=5 which maxmize the difference between task and baseline as ers frequency range.
So ers will only consider frequency range from ers_wind[0] to ers_wind[1].
Same as erd_wind.

When slide the window, erd will start from begin frequence to erd_end_f=30 HZ, and ers will start from ers_start_f=50Hz till end frq.

Reason: ERS very week compare to ERD. ERS will vanish when mean across large freq range because there are relative short large ERD between ERS.

Usage: plot tf and ers/erd for all channel or only the selected channels;
'''
import sys

import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

import os, re
import hdf5storage
import numpy as np
from scipy.io import savemat
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet
from gesture.preprocess.utils import getIndex
from common_dl import set_random_seeds
import matplotlib as mpl
from gesture.config import *
from gesture.preprocess.chn_settings import get_channel_setting
from gesture.feature_selection.utils import get_good_sids, get_final_good_sids, get_selected_channel_gumbel, \
    get_selected_channel_stg, get_selected_channel_gumbel_2_steps

seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)

sid=10
if len(sys.argv)>1:
    sid = int(float(sys.argv[1]))
fs=1000
aMovementType=0
channel_num_selected=10
pick_channel_methods='all' # 1:tf_ana, 2:all, 3:stg, 4:gumbel, 5:False
if pick_channel_methods=='tf_ana':
    selected_channels=active_channels[str(sid)]
    plot_dir = data_dir + 'tf_ersd/P' + str(sid) + '/'
elif pick_channel_methods=='stg':
    selected_channels, acc = get_selected_channel_stg(sid)
    plot_dir = data_dir + 'selection/tf_ana/'+pick_channel_methods+'/'
elif pick_channel_methods=='gumbel':
    #selected_channels, acc = get_selected_channel_gumbel(sid,10)
    selected_channels, test_acc = get_selected_channel_gumbel_2_steps(sid, channel_num_selected)
    plot_dir = data_dir + 'selection/tf_ana/gumbel/P' + str(sid)+ '/10channels/'
elif pick_channel_methods=='all':
    plot_dir = data_dir + 'tf_ersd/P' + str(sid) + '/'
else:
    selected_channels=[164, 165, 166, 167]

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

decim=4
new_fs=1000/decim

data_path = data_dir+'preprocessing/'+'P'+str(sid)+'/preprocessing2.mat'
mat=hdf5storage.loadmat(data_path)
data = mat['Datacell']
good_channels=mat['good_channels']
channelNum=len(np.squeeze(good_channels))
# total channel = channelNum + 4(2*emg + 1*trigger_indexes + 1*emg_trigger)
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
# Epoch from 4s before(idle) until 4s after(movement) stim1.
raw=raw.pick(["seeg"])
epochs = mne.Epochs(raw, events1, tmin=-4, tmax=6,baseline=None)
# or epoch from 0s to 4s which only contain movement data.
# epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

epoch1=epochs[str(aMovementType)] # 20 trials. 8001 time points per trial for 8s.
chnNum=len(epoch1.ch_names)
if pick_channel_methods=='all':
    selected_channels=range(chnNum)
## frequency analysis
# define frequencies of interest (log-spaced)
n_cycles_mthod='stage' # or: equal
fMin,fMax=2,150
fstep=1
freqs=np.arange(fMin,fMax,fstep) #148
fNum=freqs.shape[0]
#freqs = np.linspace(fMin,fMax, num=fNum)
cycleMin,cycleMax=8,50
cycleNum=fNum
#n_cycles = np.linspace(cycleMin,cycleMax, num=cycleNum)  # different number of cycle per frequency
groups=5
rates=[2,2.5,3,4,5]
num_per_group=int(fNum/groups)
if n_cycles_mthod=='equal':
    n_cycles=freqs
elif n_cycles_mthod=='stage':
    n_cycles=[]
    for g in range(groups):
        if g < groups -1:
            tmp=[int(i) for i in freqs[g*num_per_group:(g+1)*num_per_group]/rates[g]]
        elif g==groups -1:
            tmp = [int(i) for i in freqs[g * num_per_group:] / rates[g]]
        n_cycles.extend(tmp)

vmin=-4
vmax=4

# no crop here, will eliminate edge artifact later.
# different movement should have different baseline
baseline = [int(0*new_fs), int(3.5*new_fs)]
taskline=[int(4*new_fs), int(9*new_fs)]

erd_wind=10
ers_wind=2
erd_end_f=30
ers_start_f=30
erds_change = []
ch_power_avg = []

for chIndex,chID in enumerate(selected_channels):
    erds_change.append([])
    ch_power_avg.append([])
    print('Computing TF on ' + str(chIndex) + '/'+str(len(selected_channels))+' channel.')

    # decim will decrease the sfreq, so 15s will becomes 5s afterward.
    tmp = tfr_morlet(epoch1, picks=[chID],freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True,
                                     decim=decim, n_jobs=1)
    ch_power_avg[chIndex] = np.squeeze(tmp.data) #(148, 2501)
    # The very big artifact at begainning and end of the TF will cause ERS vanish after normalization.
    ch_power=np.squeeze(tfr_morlet(epoch1, picks=[chID], freqs=freqs, n_cycles=n_cycles,use_fft=True,return_itc=False, average=False,
                                   decim=decim, n_jobs=1).data) # (20, 148, 2501)
    # eliminate the artifact at the begining and end of the tf matrix
    edge=20
    #ch_power[:, :, :edge] = ch_power[:, :, edge][:,:,None]
    #ch_power[:, :, -edge:] = ch_power[:, :, -edge][:,:,None]

    base_avg = ch_power_avg[chIndex][:, baseline[0]:baseline[1]]
    base_avg_mean = np.mean(base_avg, 1) #(148,)
    base_avg_std = np.std(base_avg, 1) # (148,)

    base = ch_power[:,:, baseline[0]:baseline[1]]
    basemean = np.mean(base, 2) # (20, 148)
    basestd= np.std(base, 2)

    #Method: z-score
    #ch_power = ch_power-basemean[:, :, np.newaxis]
    #ch_power = ch_power/basestd[:, :, np.newaxis]
    # ch_power_avg[chIndex][movement] = ch_power_avg[chIndex][movement] - base_avg_mean[:, None]
    # ch_power_avg[chIndex][movement] = ch_power_avg[chIndex][movement] / base_avg_std[:, None]

    #Method: db is better
    ch_power_avg[chIndex] = 10 * np.log10(ch_power_avg[chIndex] / base_avg_mean[:, None])
    ch_power_norm = 10*np.log10(ch_power / basemean[:, :, None]) #(20, 148, 2501)

    #mean across task duration
    mean1=np.mean(ch_power_norm[:,:,taskline[0]:taskline[1]], axis=2) # (20, 148)
    mean2=np.mean(mean1,axis=0) #(148,)

    erd_start_f_index=getIndex(fMin, fMax, fstep, fMin)
    erd_end_f_index = getIndex(fMin, fMax, fstep, erd_end_f) # 30

    ers_start_f_index = getIndex(fMin, fMax, fstep, ers_start_f) # 50
    ers_end_f_index = getIndex(fMin, fMax, fstep, fMax)

    # moving window average
    erd_wind_avg = np.convolve(mean2[erd_start_f_index:erd_end_f_index], np.ones(erd_wind) / erd_wind, mode='valid')
    ers_wind_avg = np.convolve(mean2[ers_start_f_index:ers_end_f_index], np.ones(ers_wind) / ers_wind,mode='valid')
    # find the max active frequency
    erd_f1=erd_wind_avg.argmin(axis=0)
    erd_f2=erd_f1 + erd_wind
    ers_f1 = ers_start_f_index+ers_wind_avg.argmax(axis=0)
    ers_f2 = ers_start_f_index+ers_f1 + ers_wind

    #Todo: why are ERS below 0?
    move_up=2
    erd = np.mean(ch_power_norm[:, erd_f1:erd_f2, :], axis=1)+move_up # (20, 2501)
    ers = np.mean(ch_power_norm[:, ers_f1:ers_f2, :], axis=1)+move_up
    #compare_with[band] = np.mean(np.mean(ch_power_norm[:,index1:index2,:],axis=1)[:,baseline[movement][0]:baseline[movement][1]],axis=1)
    #devide by compare_with[band][:,np.newaxis] will cause result very unstable.
    erds_change[chIndex]=[]
    erds_change[chIndex].append(erd)# - compare_with[band][:,np.newaxis] # result is better if not subtract compare_with.
    erds_change[chIndex].append(ers)

import seaborn as sns
import matplotlib.gridspec as gridspec
print('Plot out to '+ plot_dir+ '.')
clrs = sns.color_palette("husl", 5)
fig = plt.figure(figsize=(6,4))

tickAndVertical=taskline
xlabels=['0s','5s']
#choose_bands_to_plot=range(len(fbands))
name_your_band=['beta','gamma']
#fig, ax=plt.subplots(2,2,sharex=True,sharey=True, squeeze=True)
for channel,chnID in enumerate(selected_channels):
    if channel == int(len(selected_channels)/2):
        print('Half way through.')
    outer = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.2)
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[0], wspace=0.01, hspace=0.01)
    # plot 2D TF
    ax1 = plt.Subplot(fig, inner[0])
    im = ax1.imshow(ch_power_avg[channel], origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax1.set_aspect('auto')
    fig.add_subplot(ax1)
    bottom_side = ax1.spines["bottom"]
    bottom_side.set_visible(False)
    ax1.axes.xaxis.set_visible(False)
    ax1.set_title('Channel '+str(chnID))
    ax1.set_ylabel('Frequency')

    for x_value in tickAndVertical:
        ax1.axvline(x=x_value,linestyle='--')

    # plot ERDS
    ax2 = plt.Subplot(fig, inner[1])
    fig.add_subplot(ax2)
    ax2.sharex(ax1)
    top_side = ax2.spines["top"]
    top_side.set_visible(False)
    #with sns.axes_style("darkgrid"):
    for band in np.arange(2):
        mean = np.mean(erds_change[channel][band],axis=0)
        times=np.arange(len(mean))
        sdt = np.std(erds_change[channel][band], axis=0)
        ax2.plot(times,mean)
        ax2.fill_between(times,mean-sdt, mean+sdt ,alpha=0.3, facecolor=clrs[band])
    #ax2.legend(name_your_band)
    ax2.set_ylabel('Change %')
    ax2.set_xticks(tickAndVertical)
    ax2.set_xticklabels(xlabels, rotation=0, ha='right', fontsize=10)
    ax2.tick_params(axis='x', which='major', pad=0.2)
    for x_value in tickAndVertical:
        ax2.axvline(x=x_value,linestyle='--')

    ax2.legend(['ERD','ERS'],loc="lower left", bbox_to_anchor=(0,0,1, 1),fontsize='small')
    cbaxes = fig.add_axes([0.93, 0.25, 0.01, 0.5]) #  [left, bottom, width, height]
    #cb = plt.colorbar(ax1, cax=cbaxes)
    fig.colorbar(im, orientation="vertical", fraction=0.046, pad=0.02, cax=cbaxes)
    filename = plot_dir + 'ERSD_'+str(sid)+'_'+str(chnID) + '.png'
    fig.savefig(filename, dpi=400)
    fig.clear()








