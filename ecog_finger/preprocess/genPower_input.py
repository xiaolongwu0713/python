import scipy.io
import mne
from ecog_finger.config import *
from ecog_finger.preprocess.chn_settings import  get_channel_setting
from dsp.common_dsp import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


try:
    mne.set_config('MNE_LOGGING_LEVEL','ERROR')
except TypeError as err:
    print(err)

sid=2
fs=1000
use_active_only=False
if use_active_only:
    active_chn=get_channel_setting(sid)
else:
    active_chn='all'

filename=data_dir+'fingerflex/data/'+str(sid)+'/'+str(sid)+'_fingerflex.mat'
mat=scipy.io.loadmat(filename)
data=mat['data'] # (46, 610040)
data=data[:,:-1]

data=np.transpose(data)
chn_num=data.shape[0]
flex=np.transpose(mat['flex']) #(5, 610040)
cue=np.transpose(mat['cue']) # (1, 610040)
data=np.concatenate((data,cue),axis=0) # (47, 610040) / (47, 610040)

chn_names=np.append(["ecog"]*chn_num,["stim"])  #,"thumb","index","middle","ring","little"])
chn_types=np.append(["ecog"]*chn_num,["stim"])  #, "emg","emg","emg","emg","emg"])
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
raw = mne.io.RawArray(data, info)

events = mne.find_events(raw, stim_channel='stim')
events=events-[0,0,1] #(150, 3)
raw=raw.pick(picks=['ecog'])

epochs = mne.Epochs(raw, events, tmin=0, tmax=2,baseline=None)

epoch1=epochs['0'].load_data() # 20 trials. 8001 time points per trial for 8s.
epoch2=epochs['1'].load_data()
epoch3=epochs['2'].load_data()
epoch4=epochs['3'].load_data()
epoch5=epochs['4'].load_data()
list_of_epochs = [epoch1, epoch2, epoch3, epoch4, epoch5]

bands_name=['theta','alpha','beta1','beta2','gamma1','gamma2','gamma3']
bandEpochs=[] # bandEpochs[0]=[deltaEpoch,thetaEpoch,....]
for fingeri in range(5):
    print('Bandpass epoch '+str(fingeri))
    bandEpochs.append([])
    thetaEpoch = list_of_epochs[fingeri].copy().pick(picks=['ecog']).filter(l_freq=fbands2['theta'][0], h_freq=fbands2['theta'][1])  # (19, 648081)
    alphaEpoch = list_of_epochs[fingeri].copy().pick(picks=['ecog']).filter(l_freq=fbands2['alpha'][0], h_freq=fbands2['alpha'][1])
    beta1Epoch = list_of_epochs[fingeri].copy().pick(picks=['ecog']).filter(l_freq=fbands2['beta1'][0],h_freq=fbands2['beta1'][1])
    beta2Epoch = list_of_epochs[fingeri].copy().pick(picks=['ecog']).filter(l_freq=fbands2['beta2'][0],h_freq=fbands2['beta2'][1])
    gamma1Epoch = list_of_epochs[fingeri].copy().pick(picks=['ecog']).filter(l_freq=fbands2['gamma1'][0],h_freq=fbands2['gamma1'][1])
    gamma2Epoch = list_of_epochs[fingeri].copy().pick(picks=['ecog']).filter(l_freq=fbands2['gamma2'][0],h_freq=fbands2['gamma2'][1])
    gamma3Epoch = list_of_epochs[fingeri].copy().pick(picks=['ecog']).filter(l_freq=fbands2['gamma3'][0],h_freq=fbands2['gamma3'][1])

    bandEpochs[fingeri] = [thetaEpoch, alphaEpoch, beta1Epoch, beta2Epoch, gamma1Epoch,gamma2Epoch,gamma3Epoch]


# Apply hilbert
print("Applying hilbert.")
for fingeri in range(5):
    for band in range(len(fbands2)):
        bandEpochs[fingeri][band].apply_hilbert(picks=['ecog'],envelope=True)

print("Stacking all 5 bands together.")
for fingeri in range(5): # change the ch_name, otherwise no way to concatenate
    for band in range(len(bands_name)):
        for c in bandEpochs[fingeri][band].ch_names:
            bandEpochs[fingeri][band].rename_channels({c: bands_name[band] + '_' + c})

rawAndBandEpochs=[]
allBandEpochs=[]
for fingeri in range(5):
    rawAndBandEpochs.append([])
    allBandEpochs=bandEpochs[fingeri][0].add_channels([bandEpochs[fingeri][i] for i in range(len(bands_name))[1:]],force_update_info=True)
    rawAndBandEpochs[fingeri]=allBandEpochs.add_channels([list_of_epochs[fingeri]],force_update_info=True)

# sanity check
trials=rawAndBandEpochs[0].get_data(picks=[-4,]) #(40, 1, 15001)
#plt.plot(trials[0,0,:])

#Save epochs
print('Saving all data.')
save_to=data_dir+'fingerflex/data/'+str(sid)+'/'
for fingeri in range(5):
    rawAndBandEpochs[fingeri].save(save_to+'rawBandEpoch'+str(fingeri)+'.fif', overwrite=True)

#a=mne.read_epochs('/Volumes/Samsung_T5/seegData/PF2/data/moveAndBandEpoch0.fif', preload=False)
#a.load_data()
#b=a.copy().pick(picks=[*range(-10,-1)])
#c=b.get_data()
##fig,ax=plt.subplots()
#ax.plot(c[0,-2,:])



