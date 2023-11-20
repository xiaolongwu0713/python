#%cd /content/drive/MyDrive/
# raw_data is imported from global config

#%%capture
#! pip install hdf5storage
#! pip install mne==0.23.0
#! pip install torch
#! pip install tensorflow-gpu == 1.12.0
#! pip install Braindecode==0.5.1

import os, re
import hdf5storage
import numpy as np
from scipy.io import savemat
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from braindecode.datautil import (create_from_mne_raw, create_from_mne_epochs)
import torch
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from braindecode.models import ShallowFBCSPNet,EEGNetv4,Deep4Net
from gesture.models.deepmodel import deepnet,deepnet_resnet
from gesture.models.EEGModels_tf import DeepConvNet_210519_512_10
from gesture.models.tsception import TSception

from gesture.myskorch import on_epoch_begin_callback, on_batch_end_callback
from gesture.config import *
from gesture.preprocess.chn_settings import get_channel_setting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import inspect as i
import sys
sys.stdout.write(i.getsource(deepnet))

pn=10 #4
Session_num,UseChn,EmgChn,TrigChn, activeChn = get_channel_setting(pn)
#fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == pn][0]
fs=1000

[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == pn][0]

loadPath = data_dir+'preprocessing'+'/P'+str(pn)+'/preprocessing2.mat'
mat=hdf5storage.loadmat(loadPath)
data = mat['Datacell']
channelNum=int(mat['channelNum'][0,0])
data=np.concatenate((data[0,0],data[0,1]),0)
del mat
# standardization
# no effect. why?
chn_data=data[:,-3:]
data=data[:,:-3]
scaler = StandardScaler()
scaler.fit(data)
data=scaler.transform((data))
data=np.concatenate((data,chn_data),axis=1)

# stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
chn_names=np.append(["seeg"]*len(UseChn),["stim0", "emg","stim1"])
chn_types=np.append(["seeg"]*len(UseChn),["stim", "emg","stim"])
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
raw = mne.io.RawArray(data.transpose(), info)


# gesture/events type: 1,2,3,4,5
events0 = mne.find_events(raw, stim_channel='stim0')
events1 = mne.find_events(raw, stim_channel='stim1')
# events number should start from 0: 0,1,2,3,4, instead of 1,2,3,4,5
events0=events0-[0,0,1]
events1=events1-[0,0,1]

#print(events[:5])  # show the first 5
# Epoch from 4s before(idle) until 4s after(movement) stim1.
raw=raw.pick(["seeg"])
epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)
# or epoch from 0s to 4s which only contain movement data.
# epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

epoch1=epochs['0'] # 20 trials. 8001 time points per trial for 8s.
epoch2=epochs['1']
epoch3=epochs['2']
epoch4=epochs['3']
epoch5=epochs['4']
list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]

#note: windows_datasets is of class BaseConcatDataset. windows_datasets.datasets is a list of all
# trials (like an epoch but organized as a list) epoched from a run.
#windows_datasets.datasets[0].windows is an epoch again created by a sliding window from one trial.


# 20 trials/epoch * 5 epochs =100 trials=100 datasets
# 1 dataset can be slided into ~161(depends on wind_size and stride) windows.
windows_datasets = create_from_mne_epochs(
    list_of_epochs,
    window_size_samples=500,
    window_stride_samples=250,
    drop_last_window=False
)


# train/valid/test split based on description column
desc=windows_datasets.description
desc=desc.rename(columns={0: 'split'})
trials_per_epoch=epoch1.events.shape[0] # 20 trial per epoch list/class
import random
val_test_num=2 # two val and two test trials
random_index = random.sample(range(trials_per_epoch), val_test_num*2)
sorted(random_index)
val_index=[rand+iclass*20 for iclass in range(5) for rand in sorted(random_index[:2]) ]
test_index=[rand+iclass*20 for iclass in range(5) for rand in sorted(random_index[-2:])]
train_index=[item for  item in list(range(100)) if item not in val_index+test_index]
desc.iloc[val_index]='validate'
desc.iloc[test_index]='test'
desc.iloc[train_index]='train'
# make sure there are val_test_num trials from each epoch (5 intotal) for both validate and test dataset
assert desc[desc['split'] == 'validate'].size == desc[desc['split'] == 'test'].size == val_test_num*5
windows_datasets.description=desc
splitted = windows_datasets.split('split')

train_set = splitted['train']
valid_set = splitted['validate']
test_set = splitted['test']

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 5
# Extract number of chans and time steps from dataset
one_window=windows_datasets.datasets[0].windows.get_data()
n_chans = one_window.shape[1]
input_window_samples = one_window.shape[2]

#model = ShallowFBCSPNet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',) # 51%
#model = EEGNetv4(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
#model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',) #87%

#expand=True/False with 4 blocks: 69%; no block 4(conv_channels=all64):78% ; no block 4(conv_channels=64,50505050):76%
#model = deepnet_resnet(n_chans,n_classes,input_window_samples=input_window_samples,expand=False) 


#model=TSception(1000,n_chans,3,3,0.5)
# Send model to GPU
if cuda:
    model.cuda()




# These values we found good for shallow network:
lr = 0.0001
weight_decay = 1e-10
batch_size = 32
n_epochs = 100

location=os.getcwd()
if re.compile('/Users/long/').match(location):
    my_callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ('on_epoch_begin_callback', on_epoch_begin_callback),('on_batch_end_callback',on_batch_end_callback),
    ]
elif re.compile('/content/drive').match(location):
   my_callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,  #torch.nn.NLLLoss/CrossEntropyLoss
    optimizer=torch.optim.Adam, #optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation; None means no validate:both train and test on training dataset.
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=my_callbacks,
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)
