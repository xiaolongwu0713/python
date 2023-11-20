#%cd /content/drive/MyDrive/
# raw_data is imported from global config

#%%capture
#! pip install hdf5storage
#! pip install mne==0.23.0
#! pip install torch==1.7.0
#! pip install Braindecode==0.5.1
#! pip install timm

'''
2s task, 2s rest.
'''
import scipy.io
import mne
import torch
from torch import nn
import timm
from braindecode import EEGClassifier
from braindecode.datautil import create_from_mne_epochs
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from sklearn.preprocessing import StandardScaler
from common_dl import set_random_seeds
from dsp.common_dsp import *
from myskorch import on_epoch_begin_callback, on_batch_end_callback
from ecog_finger.config import *
from ecog_finger.preprocess.chn_settings import  get_channel_setting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


try:
    mne.set_config('MNE_LOGGING_LEVEL','ERROR')
except TypeError as err:
    print(err)

seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)

#sys.stdout.write(i.getsource(deepnet))

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
chn_num=data.shape[0]
even_channel=1
if even_channel:
    if chn_num%2:# even channels
        pass
    else:
        data=np.concatenate((data, np.expand_dims(data[:,-1],axis=1)),axis=1)


data.shape



if 1==1:
    scaler = StandardScaler()
    scaler.fit(data)
    data=scaler.transform((data))
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
events=events-[0,0,1]
raw=raw.pick(picks=['ecog'])

tmin=0
tmax=2
if 1==1:
    event1=events[(events[:,2]==0)]
    event2=events[(events[:,2]==1)]
    event3=events[(events[:,2]==2)]
    event4=events[(events[:,2]==3)]
    event5=events[(events[:,2]==4)]

    epoch1=mne.Epochs(raw, event1, tmin=tmin, tmax=tmax,baseline=None) # 1s rest + 2s task + 1s rest
    epoch2=mne.Epochs(raw, event2, tmin=tmin, tmax=tmax,baseline=None)
    epoch3=mne.Epochs(raw, event3, tmin=tmin, tmax=tmax,baseline=None)
    epoch4=mne.Epochs(raw, event4, tmin=tmin, tmax=tmax,baseline=None)
    epoch5=mne.Epochs(raw, event5, tmin=tmin, tmax=tmax,baseline=None)

    list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]
else:
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax,baseline=None)
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


# 30 trials/epoch * 5 epochs =100 trials=150 datasets
# 1 dataset can be slided into ~161(depends on wind_size and stride) windows.
wind=500
stride=50
windows_datasets = create_from_mne_epochs(
    list_of_epochs,
    window_size_samples=wind,
    window_stride_samples=stride,
    drop_last_window=False
)



# train/valid/test split based on description column
desc=windows_datasets.description
desc=desc.rename(columns={0: 'split'})
trials_per_epoch=epoch1.events.shape[0] # 30 trial per epoch list/class
import random
val_test_num=2 # two val and two test trials/per finger
random_index = random.sample(range(trials_per_epoch), val_test_num*2)
sorted(random_index)
val_index=[ rand+iclass*30 for iclass in range(5) for rand in sorted(random_index)[:2] ]
test_index=[ rand+iclass*30 for iclass in range(5) for rand in sorted(random_index)[-2:] ]
train_index=[ item for item in list(range(150)) if item not in val_index+test_index ]
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


class_number = 5
# Extract number of chans and time stamps from dataset
one_window=windows_datasets.datasets[0].windows.get_data()
chn_num = one_window.shape[1]
input_window_samples = one_window.shape[2]

#model = ShallowFBCSPNet(chn_num,class_number,input_window_samples=input_window_samples,final_conv_length='auto',) # 51%
#model = EEGNetv4(chn_num,class_number,input_window_samples=input_window_samples,final_conv_length='auto',)

#model = deepnet(chn_num,class_number,input_window_samples=input_window_samples,final_conv_length='auto',) # 85%

#model = deepnet_resnet(chn_num,class_number,input_window_samples=input_window_samples,expand=True) # 50%

#model=d2lresnet() # sid=1: 50%
img_size=[chn_num,wind]
model = timm.create_model('visformer_tiny',num_classes=class_number,in_chans=1,img_size=img_size)

#model=TSception(208)
if cuda:
    model.cuda()

class mynet(nn.Module):
    def __init__(self, submodel):
        super().__init__()
        self.submodel=submodel
    def forward(self, x):
        x=torch.unsqueeze(x,dim=1)
        y=self.submodel(x)
        return y


net=mynet(model)

# These values we found good for shallow network:
lr = 0.0001
weight_decay = 1e-10
batch_size = 32
n_epochs = 200

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
    net,
    #criterion=torch.nn.NLLLoss,  #torch.nn.NLLLoss/CrossEntropyLoss
    criterion=torch.nn.CrossEntropyLoss,
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


