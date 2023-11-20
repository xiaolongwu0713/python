import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/我的云端硬盘/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

import hdf5storage
import numpy as np
from gesture.models.selectionModels_gumble import selectionNet
from sklearn.preprocessing import StandardScaler
import random
from comm_utils import slide_epochs
from common_dl import myDataset, cuda
from torch.utils.data import DataLoader
from gesture.config import *
import matplotlib.pyplot as plt
import torch
import glob

#syncing problem
sid=34 #4
class_number=5
channel_to_select=10
fs=1000

if socket.gethostname() == 'workstation' or socket.gethostname() == 'DESKTOP-NP9A9VI':
    if len(sys.argv)>=2: # command line
        sid=int(sys.argv[1])
        channel_to_select=int(sys.argv[2])
print("Testing on "+str(sid)+'.')
result_dir=result_dir+'selection/gumbel/'+'P'+str(sid)+'/'+'channels'+str(channel_to_select)+'/'
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
stride=50
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

train_size=len(train_loader.dataset)
val_size=len(val_loader.dataset)
test_size=len(test_loader.dataset)

# These values we found good for shallow network:
#lr = 0.0001
weight_decay = 1e-10
batch_size = 32
n_epochs = 200

#one_window.shape : (208, 500)

# Extract number of chans and time steps from dataset
one_window=next(iter(train_set))[0]
n_chans = one_window.shape[0]

img_size=[n_chans,wind]
#net = timm.create_model('visformer_tiny',num_classes=n_classes,in_chans=1,img_size=img_size)
#net = deepnet(n_chans,class_number,input_window_samples=wind,final_conv_length='auto',) # 81%
net = selectionNet(n_chans,class_number,wind,channel_to_select) # 81%
if cuda:
    net.cuda()
result_file = result_dir + '/checkpoint*.pth'
path_file=os.path.normpath(glob.glob(result_file)[0])
decoding_accuracy={}
checkpoint=torch.load(path_file)
net.load_state_dict(checkpoint['net'])
epoch=checkpoint['epoch']

running_loss = 0.0
running_corrects = 0
with torch.no_grad():
    net.eval()
    # print("Validating...")
    with torch.no_grad():
        for _, (test_x, test_y) in enumerate(test_loader):
            if (torch.cuda.is_available()):
                test_x = test_x.float().cuda()
                # val_y = val_y.float().cuda()
            else:
                test_x = test_x.float()
                # val_y = val_y.float()
            outputs = net(test_x)
            # _, preds = torch.max(outputs, 1)
            preds = outputs.argmax(dim=1, keepdim=True)

            running_corrects += torch.sum(preds.cpu().squeeze() == test_y.squeeze())
    test_acc = running_corrects.double() / test_size
    print("Testing accuracy: %{:.2f}.".format(100*test_acc))

test_acc_file=result_dir+'test_acc.npy'
np.save(test_acc_file,test_acc)


