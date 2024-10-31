import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
elif socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from gesture.decoding_ml.fbcsp.MLEngine import MLEngine

import os, re
import hdf5storage
from common_dl import set_random_seeds
from gesture.config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)

if len(sys.argv)>2:
    sid = int(float(sys.argv[1]))
    fs = int(float(sys.argv[2]))
    try:
        depth=int(float(sys.argv[6]))
        print("Depth: "+ str(depth))
    except IndexError:
        pass
else: # debug in IDE
    sid=3
    fs=1000

result_dir=data_dir+'training_result/machineLearning/FBCSP/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
class_number=5
data_path = data_dir+'preprocessing/'+'P'+str(sid)+'/preprocessing2.mat'
mat=hdf5storage.loadmat(data_path)
data = mat['Datacell']
#channelNum=int(mat['channelNum'][0,0])
good_channels = mat['good_channels']
channelNum = len(np.squeeze(good_channels))
# total channel = channelNum + 4(2*emg + 1*trigger_indexes + 1*emg_trigger)
data=np.concatenate((data[0,0],data[0,1]),0) # (1052092, 212)
del mat
data=data.transpose()# (212, 1052092)

# standardization
# no effect. why?
if 1==1: #(n_samples, n_features)
    chn_data=data[-4:,:]
    data=data[:-4,:]
    scaler = StandardScaler()
    scaler.fit(data)
    data=scaler.transform(data)
    data=np.concatenate((data,chn_data),axis=0)

# stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
chn_names=np.append(["seeg"]*channelNum,["emg0","emg1","stim_trigger","stim_emg"])
chn_types=np.append(["seeg"]*channelNum,["emg","emg","stim","stim"])
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
raw = mne.io.RawArray(data, info)


# gesture/events type: 1,2,3,4,5
events0 = mne.find_events(raw, stim_channel='stim_trigger')
events1 = mne.find_events(raw, stim_channel='stim_emg')
# events number should start from 0: 0,1,2,3,4, instead of 1,2,3,4,5
events0=events0-[0,0,1]
events1=events1-[0,0,1]

#print(events[:5])  # show the first 5
# Epoch from 4s before(idle) until 4s after(movement) stim1.
raw=raw.pick(["seeg"])
#raw=raw.pick(list(np.arange(162,168)))


if fs==1000:
    epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None).load_data().resample(500)
else:
    epochs = mne.Epochs(raw, events1, tmin=0, tmax=4, baseline=None).load_data()


# or epoch from 0s to 4s which only contain movement data.
# epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

epoch1=epochs['0'].get_data() # 20 trials. 8001 time points per trial for 8s.
epoch2=epochs['1'].get_data()
epoch3=epochs['2'].get_data()
epoch4=epochs['3'].get_data()
epoch5=epochs['4'].get_data()
list_of_epoch=np.concatenate((epoch1,epoch2,epoch3,epoch4,epoch5),axis=0) # (100, 208, 4001)

list_of_labes=[]
for i in range(5):
    trialNum=epoch1.shape[0]
    label=[[i,]*trialNum]
    list_of_labes.append(label)
list_of_labes=np.squeeze(np.asarray(list_of_labes))
list_of_labes=np.squeeze(list_of_labes.reshape((1,-1))) # (100,)

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(list_of_epoch,list_of_labes,test_size=0.3,random_state=222) # (70, 208, 4001)

dataset_details={
    'data_path' : "/Volumes/Samsung_T5/data/BCI_competition/BCICIV_2a_gdf",
    'file_to_load': 'A01T.gdf',
    'ntimes': 1,
    'kfold':5,
    'm_filters':2,
    #'window_details':{'tmin':0.0,'tmax':4.0},
    'X_train':list_of_epoch,
    'y_train':list_of_labes
}

ML_experiment = MLEngine(**dataset_details)
test_acc=ML_experiment.experiment()
print('Testing result: '+str(test_acc)+'.')
result_file=result_dir+str(sid)
np.save(result_file,test_acc)

