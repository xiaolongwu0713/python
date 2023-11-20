'''
2s task, 2s rest.
'''
import scipy.io
import mne
import torch
from braindecode import EEGClassifier
from braindecode.datautil import create_from_mne_epochs
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from common_dl import set_random_seeds
from dsp.common_dsp import *
from gesture.models.d2l_resnet import d2lresnet
from myskorch import on_epoch_begin_callback, on_batch_end_callback
from ecog_finger.config import *

sid=1
fs=1000

filename=data_dir+'fingerflex/data/'+str(sid)+'/1_fingerflex.mat'
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

tmin=-1
tmax=3
epoch1=mne.Epochs(raw, event1, tmin=tmin, tmax=tmax,baseline=None) # 1s rest + 2s task + 1s rest
epoch2=mne.Epochs(raw, event2, tmin=tmin, tmax=tmax,baseline=None)
epoch3=mne.Epochs(raw, event3, tmin=tmin, tmax=tmax,baseline=None)
epoch4=mne.Epochs(raw, event4, tmin=tmin, tmax=tmax,baseline=None)
epoch5=mne.Epochs(raw, event5, tmin=tmin, tmax=tmax,baseline=None)

#data1=epoch1.load_data().pick(picks=['ecog']).get_data() #(30 trial, 46 chn, 4001 times)
#data2=epoch2.load_data().pick(picks=['ecog']).get_data()
#data3=epoch3.load_data().pick(picks=['ecog']).get_data()
#data4=epoch4.load_data().pick(picks=['ecog']).get_data()
#data5=epoch5.load_data().pick(picks=['ecog']).get_data()

list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]

#note: windows_datasets is of class BaseConcatDataset. windows_datasets.datasets is a list of all
# trials (like an epoch but organized as a list) epoched from a run.
#windows_datasets.datasets[0].windows is an epoch again created by a sliding window from one trial.


# 30 trials/epoch * 5 epochs =100 trials=150 datasets
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
trials_per_epoch=epoch1.events.shape[0] # 30 trial per epoch list/class
import random
val_test_num=2 # two val and two test trials/per finger
random_index = random.sample(range(trials_per_epoch), val_test_num*2)
sorted(random_index)
val_index=[rand+iclass*30 for iclass in range(5) for rand in sorted(random_index)[:2] ]
test_index=[rand+iclass*30 for iclass in range(5) for rand in sorted(random_index)[-2:]]
train_index=[item for  item in list(range(150)) if item not in val_index+test_index]
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
set_random_seeds(seed=seed)

n_classes = 5
# Extract number of chans and time steps from dataset
one_window=windows_datasets.datasets[0].windows.get_data()
n_chans = one_window.shape[1]
input_window_samples = one_window.shape[2]

#model = ShallowFBCSPNet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',) # 51%
#model = EEGNetv4(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)

#model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',) # 85%

#model = deepnet_resnet(n_chans,n_classes,input_window_samples=input_window_samples,expand=True) # 50%

model=d2lresnet() # 92%

#model=TSception(208)

#model=TSception(1000,n_chans,3,3,0.5)
# Send model to GPU
if cuda:
    model.cuda()


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
    model,
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
