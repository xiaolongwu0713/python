import os
import random

import hdf5storage
import mne
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from lassonet import LassoNetClassifier

from gesture.config import *
#X, y = fetch_openml(name="mnist_784", return_X_y=True)
#filter = np.isin(y,["5","6"]) #y.isin(["5", "6"])
#X = X[filter] / 255
#y = LabelEncoder().fit_transform(y[filter])

###################
from comm_utils import slide_epochs
from gesture.preprocess.chn_settings import get_channel_setting

sid=10 #4
class_number=5
Session_num,UseChn,EmgChn,TrigChn, activeChan = get_channel_setting(sid)
#fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]
fs=1000

project_dir=data_dir+'preprocessing'+'/P'+str(sid)+'/'
result_dir=project_dir + 'result' + '/'
model_path=project_dir + 'pth' +'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(model_path):
    os.makedirs(model_path)

[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]

loadPath = data_dir+'preprocessing'+'/P'+str(sid)+'/preprocessing2.mat'
mat=hdf5storage.loadmat(loadPath)
data = mat['Datacell']
channelNum=int(mat['channelNum'][0,0])
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
stride=500
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
##################
#X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LassoNetClassifier(M=30, verbose=True)
path = model.path(X_train, y_train)

img = model.feature_importances_.reshape(28, 28)

plt.title("Feature importance to discriminate 5 and 6")
plt.imshow(img)
plt.colorbar()
plt.savefig("mnist-classification-importance.png")

n_selected = []
accuracy = []
lambda_ = []

for save in path:
    model.load(save.state_dict)
    y_pred = model.predict(X_test)
    n_selected.append(save.selected.sum())
    accuracy.append(accuracy_score(y_test, y_pred))
    lambda_.append(save.lambda_)

to_plot = [160, 220, 300]

for i, save in zip(n_selected, path):
    if not to_plot:
        break
    if i > to_plot[-1]:
        continue
    to_plot.pop()
    plt.clf()
    plt.title(f"Linear model with {i} features")
    weight = save.state_dict["skip.weight"]
    img = (weight[1] - weight[0]).reshape(28, 28)
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(f"mnist-classification-{i}.png")

fig = plt.figure(figsize=(12, 12))

plt.subplot(311)
plt.grid(True)
plt.plot(n_selected, accuracy, ".-")
plt.xlabel("number of selected features")
plt.ylabel("classification accuracy")

plt.subplot(312)
plt.grid(True)
plt.plot(lambda_, accuracy, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("classification accuracy")

plt.subplot(313)
plt.grid(True)
plt.plot(lambda_, n_selected, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("number of selected features")

plt.savefig("mnist-classification-training.png")
