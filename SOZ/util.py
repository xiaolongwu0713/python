import numpy as np
import math

import torch
from torch import nn
import matplotlib.pyplot as plt
from common_plot import *
from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity

from SOZ.config import *
from dsp.common_dsp import identify_noisy_channel

#### load data function
# get_cleaned_data(sid,start=0(hour),stop=1(hour))
def get_cleaned_data(sid,start=0,stop=1,more=True, down_sample=True,notched=True):
    participant=participants[sid]
    soz_channels = soz_channel_all[participant]  # ['B1-4', 'C1-5']

    #### get the soz channel index
    filename = meta_dir + 'labels/' + participant + '/' + participant + '_labels.npy'
    labels = np.load(filename, allow_pickle=True).item()
    soz_indexes = labels['soz_index']
    non_SOZ_index = labels['non_soz_index']

    if more==False:
        filename=data_dir+'preprocess/'+participant+'/'+participant+'.fif'
    elif more==True and down_sample==True:
        filename = data_dir + 'preprocess_more/' + participant + '/' + participant + '_down_sample.fif'
        if notched==True:
            filename = data_dir + 'preprocess_more/' + participant + '/' + participant + '_down_sample_notched.fif'
    raw = mne.io.read_raw_fif(filename)
    sf = int(raw.info['sfreq'])
    data = raw.get_data(start=start, stop=int(stop*sf))

    return data, sf, soz_indexes, non_SOZ_index # (81, 736246)

def get_windowed_nchannel_data(sid,wind,start, stop, more=True):
    '''
    return: non_soz, soz
    '''
    data, sf, soz_indexes, non_soz_index=get_cleaned_data(sid,start, stop, more=more)
    ratio=math.floor(len(non_soz_index)/len(soz_indexes))

    ## 1: non_soz; 2: soz
    if more==True:
        stride1=wind
    else:
        stride1=int(1*sf) # s
    stride2=math.floor(stride1/ratio) # 250

    ## sliding window
    start=int(0*sf) # discard first few seconds
    end=int(0*sf) # discard last few seconds
    total_len=data.shape[1]

    trials1 = math.floor((total_len - start - end - wind ) / stride1)
    data1 = np.zeros((len(non_soz_index), trials1, wind))  # (72, 365, 6000)
    for i in range(trials1):
        data1[:, i, :] = data[non_soz_index, (start + i * stride1):(start + i * stride1 + wind)]

    trials2 = math.floor((total_len - start - end - wind) / stride2)
    data2 = np.zeros((len(soz_indexes), trials2, wind))  # (9, 2920, 6000)
    for i in range(trials2):
        data2[:, i, :] = data[soz_indexes, (start + i * stride2):(start + i * stride2 + wind)]

    return data1, data2 # non_soz, soz

import matplotlib.pyplot as plt
def get_channelwise_data_label(sid,wind,start, stop):
    data1, data2 = get_windowed_nchannel_data(sid,wind,start, stop) # (channel,trial,time)(72, 365, 6000)
    flat_data1 = np.reshape(data1, (data1.shape[0] * data1.shape[1], -1)) # (26280, 6000)
    flat_data2 = np.reshape(data2, (data2.shape[0] * data2.shape[1], -1)) # (26280, 6000)
    #plt.plot(flat_data2[22,:])
    label1 = np.zeros(flat_data1.shape[0]).astype(np.int32).tolist()
    label2 = np.ones(flat_data2.shape[0]).astype(np.int32).tolist()
    return flat_data1, label1, flat_data2,label2

class test_net(nn.Module):
    def __init__(self,chn_num,class_num,wind):
        super().__init__()
        a=torch.ones((32,1,1,wind)) # 10s=10*1000
        self.ini_block=nn.Sequential(nn.Conv1d(1,2,kernel_size=(1,100)),nn.Dropout(p=0.5),nn.MaxPool2d((1,2)),nn.BatchNorm2d(2))
        a=self.ini_block(a) # torch.Size([32, 2, 1, 1450])

        block2 = nn.Sequential(nn.Conv1d(2,4,kernel_size=(1,100)),nn.MaxPool2d(1,2))
        block3 = nn.Sequential(nn.Conv1d(4, 8, kernel_size=(1, 100)), nn.MaxPool2d(1, 2))
        block4 = nn.Sequential(nn.Conv1d(8, 16, kernel_size=(1, 100)), nn.MaxPool2d(1, 2))
        block5 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=(1, 100)), nn.MaxPool2d(1, 2))
        block6 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=(1, 100)), nn.MaxPool2d(1, 2))
        self.blocks=nn.Sequential(block2,block3,block4,block5,block6)

        a=self.blocks(a)

        self.middle_block=nn.Sequential(nn.Dropout(p=0.5),nn.Flatten())
        output=self.middle_block(a)

        self.final_block=nn.Sequential(nn.Linear(output.shape[1],100),nn.LeakyReLU(),nn.Linear(100,1))


    def forward(self,x):
        x=x[:,None,:,:]
        y=self.ini_block(x)
        y=self.blocks(y)
        y=self.middle_block(y)
        y=self.final_block(y)
        return y