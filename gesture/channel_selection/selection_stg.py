## By using a customized 4-class data points, it achieves very high classification accuracy. It can select all informative features.
# This code doesn't need to pre-define the number of feature to be selected.

import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

import random
import hdf5storage
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from comm_utils import slide_epochs
from common_dl import myDataset
from gesture.models.stg.stg import STG
import numpy as np
import scipy.stats # for creating a simple dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from dataset import create_twomoon_dataset,create_4lines_dataset
import torch

from gesture.config import *
if os.environ.get('PYCHARM_HOSTED'):
    running_from_IDE=True
    running_from_CMD = False
else:
    running_from_CMD = True
    running_from_IDE = False

if socket.gethostname() == 'workstation' or socket.gethostname() == 'DESKTOP-NP9A9VI':
    if running_from_CMD: # command line
        sid=int(sys.argv[1])
        lam = float(sys.argv[2])
        #channel_to_select=int(sys.argv[2])
        print_this="Running from CMD on SID: "+str(sid)+", with lambda="+str(lam)+'.'
    elif running_from_IDE:
        sid=10
        lam=0.2
        print_this = "Running from IDE on SID: " + str(sid) + ", with lambda=" + str(lam) + '.'
print(print_this)
fs=1000
class_number=5
result_dir=result_dir+'selection/stg/'+str(lam)+'/P'+str(sid)+'test_freeze_onward/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

loadPath = data_dir+'preprocessing'+'/P'+str(sid)+'/preprocessing2.mat'
mat=hdf5storage.loadmat(loadPath)
data = mat['Datacell']
good_channels=mat['good_channels']
channelNum=len(np.squeeze(good_channels))
data=np.concatenate((data[0,0],data[0,1]),0)
del mat
# standardization
# no effect. why?
if 1==1:
    chn_data=data[:,-3:]
    data=data[:,:-3]
    scaler = StandardScaler()
    scaler.fit(data)
    data=scaler.transform((data))
    data=np.concatenate((data,chn_data),axis=1)

# stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
chn_names=np.append(["seeg"]*channelNum,["emg0","emg1","stim_trigger","stim_emg"])
chn_types=np.append(["seeg"]*channelNum,["emg","emg","stim","stim"])
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
epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)
# or epoch from 0s to 4s which only contain movement data.
# epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

epoch1=epochs['0'].get_data() # 20 trials. 8001 time points per trial for 8s.
epoch2=epochs['1'].get_data()
epoch3=epochs['2'].get_data()
epoch4=epochs['3'].get_data()
epoch5=epochs['4'].get_data()
list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]
total_len=list_of_epochs[0].shape[2]

# validate=test=2 trials
trial_number=[list(range(epochi.shape[0])) for epochi in list_of_epochs] #[ [0,1,2,...19],[0,1,2...19],... ]
test_trials=[random.sample(epochi, 2) for epochi in trial_number]
# len(test_trials[0]) # test trials number
trial_number_left=[np.setdiff1d(trial_number[i],test_trials[i]) for i in range(class_number)]

val_trials=[random.sample(list(epochi), 2) for epochi in trial_number_left]
train_trials=[np.setdiff1d(trial_number_left[i],val_trials[i]).tolist() for i in range(class_number)]

# no missing trials
assert [sorted(test_trials[i]+val_trials[i]+train_trials[i]) for i in range(class_number)] == trial_number

test_epochs=[epochi[test_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)] # [ epoch0,epoch1,epch2,epoch3,epoch4 ]
val_epochs=[epochi[val_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)]
train_epochs=[epochi[train_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)]


wind=500
stride=100
X_train=[]
y_train=[]
X_val=[]
y_val=[]
X_test=[]
y_test=[]

for clas, epochi in enumerate(test_epochs):
    Xi,y=slide_epochs(epochi,clas,wind, stride)
    assert Xi.shape[0]==len(y)
    X_test.append(Xi)
    y_test.append(y)
X_test=np.concatenate(X_test,axis=0) # (1300, 63, 500)
y_test=np.asarray(y_test)
y_test=np.reshape(y_test,(-1,1)) # (5, 270)

for clas, epochi in enumerate(val_epochs):
    Xi,y=slide_epochs(epochi,clas,wind, stride)
    assert Xi.shape[0]==len(y)
    X_val.append(Xi)
    y_val.append(y)
X_val=np.concatenate(X_val,axis=0) # (1300, 63, 500)
y_val=np.asarray(y_val)
y_val=np.reshape(y_val,(-1,1)) # (5, 270)

for clas, epochi in enumerate(train_epochs):
    Xi,y=slide_epochs(epochi,clas,wind, stride)
    assert Xi.shape[0]==len(y)
    X_train.append(Xi)
    y_train.append(y)
X_train=np.concatenate(X_train,axis=0) # (1300, 63, 500)
y_train=np.asarray(y_train)
y_train=np.reshape(y_train,(-1,1)) # (5, 270)
chn_num=X_train.shape[1]

train_set=myDataset(X_train,y_train)
val_set=myDataset(X_val,y_val)
test_set=myDataset(X_test,y_test)

batch_size = 32
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)
one_window=next(iter(train_set))[0]
n_chans = one_window.shape[0]
###################################################################################################
#X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.3)
#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

args_cuda = torch.cuda.is_available()
device = torch.device("cuda" if args_cuda else "cpu")

def two_stages_ascend(start_value,end_value,epochs,end_epoch):
    t = torch.FloatTensor(torch.arange(0.0,epochs))
    out=torch.zeros(epochs)
    out[:end_epoch]=start_value
    a=(end_value-start_value)/(epochs-end_epoch)
    b=end_value-epochs*a
    y=a*t[end_epoch:]+b
    out[end_epoch:]=y
    return out

epochs=200
start_to_shrink=20
lam_schedule=two_stages_ascend(0.00000001,0.5, epochs, start_to_shrink)
patients=20
task_meta={}
task_meta['task']='classification'
task_meta['result_dir']=result_dir
# freeze_onward=20; when to freeze the selection net???
model = STG(n_chans,class_number,wind, task_meta, freeze_onward=None, patients=patients, task_type='classification',batch_size=32, sigma=0.5, lam=lam, lam_schedule=lam_schedule, random_state=1, device=device)

result_prob, result_raw, train_loss, validate_acc=model.fit(train_loader, val_loader, epochs, print_interval=1, early_stop=True)
result_prob=np.asarray(result_prob)
result_raw=np.asarray(result_raw)
train_loss=np.asarray(train_loss)
validate_acc=np.asarray(validate_acc)
prob=model.get_gates(mode='prob')
raw=model.get_gates(mode='raw')
import matplotlib.pyplot as plt
#plt.plot(prob)
#plt.plot(raw)

# testing
truth, y_pred=model.predict(test_loader)
test_acc=sum(truth==y_pred)/truth.shape[0]

result={}
result['probs']=result_prob
result['raws']=result_raw
result['train_loss']=train_loss
result['validate_acc']=validate_acc
result['test_acc']=test_acc
filename=result_dir+"result"+str(sid)
np.save(filename, result)
#load_back=np.load('result10.npy', allow_pickle=True).item()




