'''
This script is to classify phonemes: a/e/i/u
'''
import glob
import matplotlib
from braindecode.models import EEGNetv4, ShallowFBCSPNet

from gesture.decoding_ml.fbcsp.MLEngine import MLEngine
from gesture.models.deepmodel import deepnet

matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from speech_pinyin_Ruijin.config import *
from dSPEECH.phoneme.util import train_test_split, wind_list_of_2D
from gesture.models.d2l_resnet import d2lresnet_simple, d2lresnet
from gesture.utils import windowed_data

#device=torch.device('cpu')
modality='SEEG'
task='pinyin'
sid=2 # 31%
sf=1000

folder=data_dir+str(sid)+'-*'
folder=os.path.normpath(glob.glob(folder)[0])
folder=folder.replace("\\", "/")
result_dir=folder+'/result/'
filename=folder+'/processed/'+task+'.fif'
raw=mne.io.read_raw_fif(filename,preload=True)
info=raw.info

# normalization helps
#raw.apply_function(lambda x: (x - np.mean(x) / np.std(x)))
if 1==1:
    stim=raw.get_data(picks=['stim'])
    data=raw.get_data(picks=['eeg',]).transpose()
    scaler = StandardScaler()
    data2 = scaler.fit_transform((data))
    raw = mne.io.RawArray(np.concatenate((data2.transpose(),stim),axis=0), info)

events=mne.find_events(raw, stim_channel='Trigger')
raw.drop_channels(['Trigger'])
events=events[3:-1,:]
events_tasks=np.asarray([tmp for tmp in events if tmp[-1] != 99])
epochs_all = mne.Epochs(raw, events_tasks, tmin=3.5, tmax=4.5,baseline=None)#.load_data().resample(500)
epoch1=epochs_all['1'].get_data() # (16, 118, 1001)
epoch2=epochs_all['2'].get_data()
epoch3=epochs_all['3'].get_data()
epoch4=epochs_all['4'].get_data()

if 1==0:
    # normalization here
    def norm_epoch(epoch):
        for i in range(len(epoch)):
            ee=epoch[i]
            tmp=ee.transpose()
            scaler = StandardScaler()
            tmp = scaler.fit_transform((tmp))
            epoch[i]=tmp.transpose()
        return epoch
    epoch1=norm_epoch(epoch1)
    epoch2=norm_epoch(epoch2)
    epoch3=norm_epoch(epoch3)
    epoch4=norm_epoch(epoch4)

list_of_epoch_tmp=[epoch1,epoch2,epoch3,epoch4]
list_of_epoch=np.concatenate((epoch1,epoch2,epoch3,epoch4),axis=0) # (100, 208, 4001)

list_of_labes=[]
for i in range(4):
    trialNum=list_of_epoch_tmp[i].shape[0]
    label=[i,]*trialNum
    list_of_labes=list_of_labes+label
list_of_labes=np.asarray(list_of_labes)
#list_of_labes=np.squeeze(np.asarray(list_of_labes))
#list_of_labes=np.squeeze(list_of_labes.reshape((1,-1))) # (100,)

#from sklearn.model_selection import train_test_split
#X_train,X_val,y_train,y_val=train_test_split(list_of_epoch,list_of_labes,test_size=0.3,random_state=222) # (70, 208, 4001)

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



