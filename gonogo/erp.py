# choose channels from tf analysis, then plot the erp ERP
# plot erp of a frequency range:[l_freq,h_freq]

import hdf5storage
import os
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet
import math
from gonogo.config import *

sid=6 #4
l_freq=0.5
h_freq=2
data_dir='/Volumes/Samsung_T5/data/ruijin/gonogo/preprocessing/P'+str(sid)
plot_dir=data_dir + '/erpPlots/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

#Session_num,UseChn,EmgChn,TrigChn = get_channel_setting(sid)
#original_fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == pn][0]
loadPath = data_dir+'/preprocessing/preprocessingv2.mat'
mat=hdf5storage.loadmat(loadPath)
fs=mat['Fs']
rtime=mat['ReactionTime']
rtime=np.concatenate((rtime[0,0],rtime[0,1]),axis=0)
data=mat['DATA']
data=np.concatenate((data[0,0],data[0,1]),axis=0) #(2160440, 63)
events=mat['Trigger']
events=np.concatenate((events[0,0],events[0,1]),axis=0) # two sessions
events[:, [1,2]] = events[:, [2,1]] # swap 1st and 2nd column to: timepoint, duration, event code
events=events.astype(int)

del mat

chn_num=data.shape[1]

event1=events[(events[:,2]==1)]
event2=events[(events[:,2]==2)]
event3=events[(events[:,2]==3)]
event4=events[(events[:,2]==4)]
event5=events[(events[:,2]==3)]
event4=events[(events[:,2]==4)]
event34_index=[i or j for (i,j) in zip((events[:,2]==3), (events[:,2]==4))]
event34=events[event34_index]
event56_index=[i or j for (i,j) in zip((events[:,2]==5), (events[:,2]==6))]
event56=events[event56_index]
event1112_index=[i or j for (i,j) in zip((events[:,2]==11), (events[:,2]==12))]
event1112=events[event1112_index]
event2122_index=[i or j for (i,j) in zip((events[:,2]==21), (events[:,2]==22))]
event2122=events[event2122_index]
event11=events[(events[:,2]==11)]
event12=events[(events[:,2]==12)]
event21=events[(events[:,2]==21)]
event22=events[(events[:,2]==22)]


chn_names=np.asarray(["seeg"]*chn_num)
chn_types=np.asarray(["seeg"]*chn_num)
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
raw = mne.io.RawArray(data.transpose(), info)

epoch1=mne.Epochs(raw, event1, tmin=-1, tmax=4,baseline=None) # fixed 3s
epoch2=mne.Epochs(raw, event2, tmin=-1, tmax=4,baseline=None) # fixed 3s
epoch34=mne.Epochs(raw, event34, tmin=-3, tmax=1,baseline=None) # varying reaction time, 3s maximum
epoch56=mne.Epochs(raw, event56, tmin=-3, tmax=1,baseline=None) # varying reaction time, 3s maximum
epoch1112=mne.Epochs(raw, event1112, tmin=-7, tmax=6,baseline=None) # 3s task cue and 3s executing cue
epoch2122=mne.Epochs(raw, event2122, tmin=0, tmax=3,baseline=None) # 3s executing cue
epoch11=mne.Epochs(raw, event11, tmin=-7, tmax=4.0,baseline=None)
epoch12=mne.Epochs(raw, event12, tmin=-7, tmax=4.0,baseline=None)
epoch21=mne.Epochs(raw, event21, tmin=-0.5, tmax=4.0,baseline=None)
epoch22=mne.Epochs(raw, event22, tmin=-0.5, tmax=4.0,baseline=None)

erp11 = epoch11.load_data().copy().pick(picks=['seeg']).filter(l_freq=l_freq, h_freq=h_freq)
erp12 = epoch12.load_data().copy().pick(picks=['seeg']).filter(l_freq=l_freq, h_freq=h_freq)
erp11_avg=erp11.average(method='mean')
erp12_avg=erp12.average(method='mean')

erp21 = epoch21.load_data().copy().pick(picks=['seeg']).filter(l_freq=l_freq, h_freq=h_freq)
erp22 = epoch22.load_data().copy().pick(picks=['seeg']).filter(l_freq=l_freq, h_freq=h_freq)
erp21_avg=erp21.average(method='mean')
erp22_avg=erp22.average(method='mean')

erps=[erp11_avg, erp12_avg, erp21_avg, erp22_avg]
erp_names=['11','12','21','22']
plotspp=12
pages=math.ceil(chn_num/plotspp)
# plot on 3*4 subplots=12/pdf
rows=3
columns=4
fig,ax=plt.subplots() # create a figure
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_xticks([])
for erpi,erp_name in zip(erps,erp_names):
    if erp_name=='11' or erp_name=='12':
        xticklabels= [0, 3, 6, 9]
        xticks = [i * 1000 for i in xticklabels]
        vlines=[3000,6000,9000]
    elif erp_name=='21' or erp_name=='22':
        xticklabels = [0.5, 3.5]
        xticks = [int(i * 1000) for i in xticklabels]
        vlines = [500,3500]
    for page in range(pages):
        ax = fig.subplots(rows, columns, sharex=True)  # partition in to ax
        for r in range(rows):
            for c in range(columns):
                if not plotspp*page+ r*columns+c >= chn_num:
                    chn_i=plotspp*page+ r*columns +c
                    axi=ax[r,c]
                    axi.yaxis.set_tick_params(labelsize=5)  # tick lable smaller
                    axi.tick_params(axis='y', which='major', pad=1)  # space between ticklabel and yaxis
                    tmp = erpi.copy().pick(picks=[chn_i]).data
                    axi.plot(tmp.transpose(),)
                    axi.text(0.4, 0.8, 'chn'+str(chn_i), fontsize=12,transform=axi.transAxes) # , transform=axi.transFigure
                    axi.set_xticklabels([])
                    axi.set_xticks([])
                    for vline in vlines:
                        axi.axvline(x=vline,linewidth=1, color='r',linestyle='--')
                # set x axis of last row
                if r==rows-1:
                    axi.set_xticks(xticks)
                    axi.set_xticklabels(xticklabels)
                    ymin=axi.get_ylim()[0]
                    if erp_name=='11' or erp_name=='12':
                        axi.text(1, ymin, 'S1-1', fontsize=8) # S1 warning
                        axi.text(3000, ymin, 'S2-1', fontsize=8) # s2 warning
                        axi.text(6000, ymin, 'S1-2', fontsize=8) # s1 imperative
                    elif erp_name == '21' or erp_name == '22':
                        axi.text(1000, ymin, 'S2-2', fontsize=8) # s2 imperative

        start_ch=page*plotspp
        end_ch=chn_i
        plt.subplots_adjust(wspace=0.15, hspace=0.03)  # or left=0.1,bottom=0.1, right=0.9, top=0.9,
        figname = plot_dir + 'erp'+erp_name+'_ch' + str(start_ch)+'-to-ch'+str(end_ch) + '.pdf'
        fig.savefig(figname)
        fig.clear()


