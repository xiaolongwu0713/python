import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
elif socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
from gesture.config import *

# Input expected by sklean: X(samples, features), Y(samples,). For example X(1000,20),y(1000,). Each sample is a 1D vector.

import hdf5storage
from common_dl import set_random_seeds
from comm_utils import slide_epochs
from sklearn.preprocessing import StandardScaler
from gesture.config import *
from gesture.preprocess.chn_settings import get_active_channel, get_channel_setting

if len(sys.argv)>2:
    sid = int(float(sys.argv[1]))
    active_channel = sys.argv[2]
else: # debug in IDE
    sid=10
    active_channel='all'

class_number=5
Session_num,UseChn,EmgChn,TrigChn = get_channel_setting(sid)
activeChan=get_active_channel(sid)
if active_channel=='all':
    pick_channel='all'
elif active_channel=='active':
    pick_channel=activeChan
else:
    raise NameError("Select channel with 'all' or 'active'")
#fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]
fs=1000

sid_dir=data_dir+'preprocessing'+'/P'+str(sid)+'/'
#[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]

loadPath = sid_dir+'preprocessing2.mat'
mat=hdf5storage.loadmat(loadPath)
data = mat['Datacell']
channelNum=int(mat['channelNum'][0,0])
data=np.concatenate((data[0,0],data[0,1]),0)
del mat
# standardization
# no effect. why?
if 1==0:
    chn_data=data[:,-3:]
    data=data[:,:-3]
    scaler = StandardScaler()
    scaler.fit(data)
    data=scaler.transform((data))
    data=np.concatenate((data,chn_data),axis=1)

# stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
chn_names = np.append(["seeg"] * channelNum, ["emg0", "emg1", "stim_trigger", "stim_emg"])
chn_types = np.append(["seeg"] * channelNum, ["emg", "emg", "stim", "stim"])
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
epochs = mne.Epochs(raw, events1, tmin=-3, tmax=4,baseline=None)
epochs=epochs.load_data().pick(picks=pick_channel) # activeChan/'all'
epoch_info=epochs.info
# or epoch from 0s to 4s which only contain movement data.
# epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

epoch1=epochs['0'] # 20 trials. 8001 time points per trial for 8s.
epoch2=epochs['1']
epoch3=epochs['2']
epoch4=epochs['3']
epoch5=epochs['4']

epoch1_bsl=epoch1.load_data().copy().crop(-3,0)
epoch2_bsl=epoch1.load_data().copy().crop(-3,0)
epoch3_bsl=epoch1.load_data().copy().crop(-3,0)
epoch4_bsl=epoch1.load_data().copy().crop(-3,0)
epoch5_bsl=epoch1.load_data().copy().crop(-3,0)
list_of_epochs_bsl=[epoch1_bsl,epoch2_bsl,epoch3_bsl,epoch4_bsl,epoch5_bsl]

epoch1=epoch1.load_data().crop(0,4).get_data()
epoch2=epoch2.load_data().crop(0,4).get_data()
epoch3=epoch3.load_data().crop(0,4).get_data()
epoch4=epoch4.load_data().crop(0,4).get_data()
epoch5=epoch5.load_data().crop(0,4).get_data()
list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]
wind=500
stride=250
for i in range(5):
    clas=i + 1
    Xi,y=slide_epochs(list_of_epochs[i],clas,wind, stride)
    tmp_epoch=mne.EpochsArray(Xi,epoch_info)
    events=tmp_epoch.events
    events[:,2]=clas
    tmp_epoch.events=events
    list_of_epochs[i]=tmp_epoch
trialN=Xi.shape[0]

list_of_epochs_bsl_psd=[]
for i in range(5):
    (tmp,freq)=mne.time_frequency.psd_welch(list_of_epochs_bsl[i],average='mean')
    tmp=np.mean(tmp,axis=0) # (208, 129)
    list_of_epochs_bsl_psd.append(tmp)

print("PSD started.")
list_of_epochs_psd=[]
for i in range(5):
    (tmp,freq)=mne.time_frequency.psd_welch(list_of_epochs[i],average='mean') # tmp: (300, 208, 129)
    #tmp=np.mean(tmp,axis=0) # (208, 129)
    list_of_epochs_psd.append(tmp)
print("PSD done.")
# average across below frequency ranges
fbands = [[1,4],[4,8],[8,13],[13,30],[60,75],[75,95],[105,125],[125,145],[155,195]]
chnN=list_of_epochs_psd[0].shape[1]
fbandsN=len(fbands)

list_of_epochs_bsl_psd_avg=[]
for i in range(5):
    tmp_epoch_bsl_psd_avg = np.zeros([chnN, fbandsN])
    for k,fi in enumerate(fbands):
        lowf=fi[0]
        highf=fi[1]
        lowf_index,_=min(enumerate(freq), key=lambda x: abs(x[1] - lowf))
        highf_index,_ = min(enumerate(freq), key=lambda x: abs(x[1] - highf))
        tmp_epoch_bsl_psd_avg[:,k]=np.mean(list_of_epochs_bsl_psd[i][:,lowf_index:highf_index],axis=1)
    list_of_epochs_bsl_psd_avg.append(tmp_epoch_bsl_psd_avg)


list_of_epochs_psd_avg=[]
for i in range(5):
    tmp_epoch_psd_avg = np.zeros([trialN,chnN, fbandsN])
    #tmp_epoch_psd_avg = np.zeros([trialN, chnN*fbandsN])
    for k,fi in enumerate(fbands):
        lowf=fi[0]
        highf=fi[1]
        lowf_index,_=min(enumerate(freq), key=lambda x: abs(x[1] - lowf))
        highf_index,_ = min(enumerate(freq), key=lambda x: abs(x[1] - highf))
        tmp_epoch_psd_avg[:,:,k]=np.mean(list_of_epochs_psd[i][:,:,lowf_index:highf_index],axis=2)
    tmp_epoch_psd_avg=tmp_epoch_psd_avg-list_of_epochs_bsl_psd_avg[i]
    tmp=np.reshape(tmp_epoch_psd_avg,(trialN, chnN*fbandsN))
    list_of_epochs_psd_avg.append(tmp)

list_of_epochs_psd_avg=np.concatenate(list_of_epochs_psd_avg,axis=0) # (1500, 23*9=207)
print("Average done/")

list_of_labes=[]
for i in range(5):
    trialN=tmp.shape[0]
    label=[[i,]*trialN]
    list_of_labes.append(label)
list_of_labes=np.squeeze(np.asarray(list_of_labes))
list_of_labes=np.squeeze(list_of_labes.reshape((1,-1))) # (1500,)

filename=sid_dir+'psd_feature'
np.save(filename,list_of_epochs_psd_avg)
filename=sid_dir+'label'
np.save(filename,list_of_labes)

# try use low frequency local motor potential as feature

