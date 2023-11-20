'''
2s task, 2s rest.
'''
import scipy.io
import mne
from mne.time_frequency import tfr_morlet

from dsp.common_dsp import *
from ecog_finger.config import *

sid=3
fs=1000

filename=data_dir+'fingerflex/data/'+str(sid)+'/'+str(sid)+'_fingerflex.mat'
#data_dir='/Volumes/Samsung_T5/data/gesture/preprocessing/P'+str(sid)
plot_dir=data_dir +'fingerflex/data/'+str(sid) +'/tfPlot/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

mat=scipy.io.loadmat(filename)
data=np.transpose(mat['data']) # (46, 610040)
chn_num=data.shape[0]
flex=np.transpose(mat['flex']) #(5, 610040)
cue=np.transpose(mat['cue']) # (1, 610040)
data=np.concatenate((data,cue,flex),axis=0) # (47, 610040) / (52, 610040)

chn_names=np.append(["ecog"]*chn_num,["stim","thumb","index","middle","ring","little"])
chn_types=np.append(["ecog"]*chn_num,["stim", "emg","emg","emg","emg","emg"])
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
raw = mne.io.RawArray(data, info)

events = mne.find_events(raw, stim_channel='stim')
events=events-[0,0,1]
'''
verify the events are picked up correctly.
a=np.asarray([i for i in events if i[2]==1])
fig,ax=plt.subplots()
ax.plot(cue[0,:111080])
for i in a[:6]:
    ax.axvline(x=i[0],linewidth=1,color='r',linestyle='--')
'''
event1=events[(events[:,2]==0)]
event2=events[(events[:,2]==1)]
event3=events[(events[:,2]==2)]
event4=events[(events[:,2]==3)]
event5=events[(events[:,2]==4)]

tmin=-2
tmax=4
epoch1=mne.Epochs(raw, event1, picks=['ecog'], tmin=tmin, tmax=tmax,baseline=None) # 1s rest + 2s task + 1s rest
epoch2=mne.Epochs(raw, event2, picks=['ecog'], tmin=tmin, tmax=tmax,baseline=None)
epoch3=mne.Epochs(raw, event3, picks=['ecog'], tmin=tmin, tmax=tmax,baseline=None)
epoch4=mne.Epochs(raw, event4, picks=['ecog'], tmin=tmin, tmax=tmax,baseline=None)
epoch5=mne.Epochs(raw, event5, picks=['ecog'], tmin=tmin, tmax=tmax,baseline=None)

#data1=epoch1.load_data().pick(picks=['ecog']).get_data() #(30 trial, 46 chn, 4001 times)
#data2=epoch2.load_data().pick(picks=['ecog']).get_data()
#data3=epoch3.load_data().pick(picks=['ecog']).get_data()
#data4=epoch4.load_data().pick(picks=['ecog']).get_data()
#data5=epoch5.load_data().pick(picks=['ecog']).get_data()

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
decim=4
new_fs=1000/decim
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
crop=50 #0.2s
shift=crop #int(crop*new_fs)
crop1=0
crop2=4
for channel in range(len(sub_ch_names)):
    power.append([])
    if shift==0:
        power=averagePower
    else:
        power[channel]=averagePower[channel][:,shift:-shift]


onset_time=int(2*new_fs) #s before crop
new_onset_time=int(2*new_fs-crop)
baseline = [int(1),int(new_onset_time-0.5*new_fs)] # crop to 0.5s before onset
tickpos=[new_onset_time,new_onset_time+int(2*new_fs)]
ticklabel=['onset','offset']

#(300, 5001)
vmin=-4
vmax=4
fig, ax = plt.subplots()
print('Ploting out to '+plot_dir+'.')
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
    #fig.colorbar(im, ax=ax0)
    fig.colorbar(im0, orientation="vertical",fraction=0.046, pad=0.02,ax=ax)

    # save
    figname = plot_dir + 'tf_compare_'+str(channel) + '.pdf'
    fig.savefig(figname)

    # clean up plotting area
    ax.images[-1].colorbar.remove()
    ax.cla()
plt.close(fig)


