import random
from torch import nn
import hdf5storage
from braindecode.models.modules import Expression
from torch.utils.data import DataLoader

from comm_utils import slide_epochs
from common_dl import myDataset
from gesture.config import *
import torch
from gesture.models.deepmodel import deepnet, deepnet_seq, deepnet_rnn, deepnet_da, swap_time_spat, swap_plan_spat, \
    deepnet_da_expandPlan, deepnet_expandPlan
from scipy import signal
from scipy.fft import fft,fft2, fftfreq
import numpy as np
import matplotlib.pyplot as plt

sid=41
model_name='deepnet'

'''
params=list(net.named_parameters())
kernels=params[0][1].squeeze().detach().numpy() # ([64, 50]), 64 kernel in total

mat=np.zeros((kernels.shape[0],25))
N=kernels.shape[1]
for idx,kernel in enumerate(kernels):
    yf = fft(kernel)
    xf = fftfreq(N, 1/1000)[:N//2]
    yf=2.0 / N * np.abs(yf[0:N // 2])
    mat[idx,:]=yf

fig,ax=plt.subplots()
vmin=0
im=ax.imshow(mat,origin='lower',cmap='RdBu_r')
filename=savefolder+'filterOf'+model_name+'.pdf'
fig.savefig(filename)
'''


# load input
class_number=5
fs=1000
wind = 500
stride = 500

#data_path = data_dir+'preprocessing/'+'P'+str(sid)+'/preprocessing2.mat'
data_path='/Users/long/Documents/BCI/python_scripts/googleDrive/data/gesture/preprocessing/P41/preprocessing2.mat'
mat=hdf5storage.loadmat(data_path)
data = mat['Datacell']
channelNum=int(mat['channelNum'][0,0])
# total channel = channelNum + 4(2*emg + 1*trigger_indexes + 1*emg_trigger)
data=np.concatenate((data[0,0],data[0,1]),0)
del mat
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

X_train=[]
y_train=[]
X_val=[]
y_val=[]
X_test=[]
y_test=[]

X=[]
y=[]
for clas, epochi in enumerate(list_of_epochs):
    Xi,yi=slide_epochs(epochi,clas,wind, stride)
    assert Xi.shape[0]==len(yi)
    X.append(Xi)
    y.append(yi)

sampleSize=[]
ds=[]
dl=[]
for i in range(5):
    sampleSize.append(X[i].shape[0])
    ds.append(myDataset(X[i],np.asarray(y[i])))
    dl.append(DataLoader(dataset=ds[i], batch_size=X[i].shape[0], shuffle=False, pin_memory=False))
chn_num=X[0].shape[1]



savefolder=data_dir+'training_result/asFilter/'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

########### test on different network #################
############# model ############
net0 = deepnet_expandPlan(chn_num,5,500)
model_result_dir='/Users/long/Documents/BCI/python_scripts/googleDrive/data/gesture/preprocessing/P41/pth/checkpoint_deepnet_expandPlan_59.pth'
checkpoint = torch.load(model_result_dir,map_location=torch.device('cpu'))
net0.load_state_dict(checkpoint['net'])

subnets0=[]
subnets0.append(net0.conv_time)
subnets0.append(nn.Sequential(subnets0[-1],net0.conv_spatial))
subnets0.append(nn.Sequential(subnets0[-1],net0.bn1,net0.nonlinear1,net0.mp1,net0.drop1))
subnets0.append(nn.Sequential(subnets0[-1],net0.conv2))
subnets0.append(nn.Sequential(subnets0[-1],net0.bn2,net0.nonlinear2,net0.drop2))
subnets0.append(nn.Sequential(subnets0[-1],net0.conv3))
subnets0.append(nn.Sequential(subnets0[-1],net0.bn3,net0.nonlinear3,net0.drop3))
subnets0.append(nn.Sequential(subnets0[-1],net0.conv4))


############# model ############
net = deepnet(chn_num,5,500)
model_result_dir='/Users/long/Documents/BCI/python_scripts/googleDrive/data/gesture/preprocessing/P41/pth/checkpoint_deepnet_18.pth'
checkpoint = torch.load(model_result_dir,map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['net'])

subnets=[]
subnets.append(net.conv_time)
subnets.append(nn.Sequential(subnets[-1],net.conv_spatial))
subnets.append(nn.Sequential(subnets[-1],net.bn1,net.nonlinear1,net.mp1,net.drop1))
subnets.append(nn.Sequential(subnets[-1],net.conv2))
subnets.append(nn.Sequential(subnets[-1],net.bn2,net.nonlinear2,net.drop2))
subnets.append(nn.Sequential(subnets[-1],net.conv3))
subnets.append(nn.Sequential(subnets[-1],net.bn3,net.nonlinear3,net.drop3))
subnets.append(nn.Sequential(subnets[-1],net.conv4))

############# model ############
net2=deepnet_da(chn_num,5,500)
model_result_dir='/Users/long/Documents/BCI/python_scripts/googleDrive/data/gesture/preprocessing/P41/pth/checkpoint_deepnet_da_43.pth'
checkpoint = torch.load(model_result_dir,map_location=torch.device('cpu'))
net2.load_state_dict(checkpoint['net'])

subnets2=[]
subnets2.append(nn.Sequential(net2.conv_spatial_top,Expression(swap_plan_spat),net2.conv_time))
#subnets2.append(nn.Sequential(subnets2[-1],nn.Sequential(Expression(swap_time_spat),net2.conv_spatial) ))
subnets2.append(nn.Sequential(subnets2[-1],net2.conv_spatial ))
subnets2.append(nn.Sequential(subnets2[-1],net2.bn1,net2.nonlinear1,net2.mp1,net2.drop1))
subnets2.append(nn.Sequential(subnets2[-1],net2.conv2))
subnets2.append(nn.Sequential(subnets2[-1],net2.bn2,net2.nonlinear2,net2.drop2))
subnets2.append(nn.Sequential(subnets2[-1],net2.conv3))
subnets2.append(nn.Sequential(subnets2[-1],net2.bn3,net2.nonlinear3,net2.drop3))
subnets2.append(nn.Sequential(subnets2[-1],net2.conv4))

############# model ############
net3=deepnet_da_expandPlan(chn_num,5,500)
model_result_dir='/Users/long/Documents/BCI/python_scripts/googleDrive/data/gesture/preprocessing/P41/pth/checkpoint_deepnet_da_expandPlan_36.pth'
checkpoint = torch.load(model_result_dir,map_location=torch.device('cpu'))
net3.load_state_dict(checkpoint['net'])

subnets3=[]
subnets3.append(nn.Sequential(net3.conv_spatial_top,Expression(swap_plan_spat),net3.conv_time))
#subnets3.append(nn.Sequential(subnets3[-1],nn.Sequential(Expression(swap_time_spat),net3.conv_spatial) ))
subnets3.append(nn.Sequential(subnets3[-1],net3.conv_spatial ))
subnets3.append(nn.Sequential(subnets3[-1],net3.bn1,net3.nonlinear1,net3.mp1,net3.drop1))
subnets3.append(nn.Sequential(subnets3[-1],net3.conv2))
subnets3.append(nn.Sequential(subnets3[-1],net3.bn2,net3.nonlinear2,net3.drop2))
subnets3.append(nn.Sequential(subnets3[-1],net3.conv3))
subnets3.append(nn.Sequential(subnets3[-1],net3.bn3,net3.nonlinear3,net3.drop3))
subnets3.append(nn.Sequential(subnets3[-1],net3.conv4))


subnets=subnets0
## feed into the network
outs=[]
for layer,subnet in enumerate(subnets):
    outs.append([])
    for i in range(5):
        [x,y]=iter(dl[i]).next() # torch.Size([160, 190, 500])
        out=subnet(x.unsqueeze(1).float())
        # out=net.ap(out)
        out = out.squeeze().detach().numpy()  # torch.Size([160, 50, 123])
        outs[layer].append(out) # torch.Size([160, 50, 123])

filterNum=out.shape[1]
fig,ax=plt.subplots()
N=out.shape[2]
colors = ['orangered', 'yellow', 'gold', 'orange', 'springgreen', 'aquamarine', 'skyblue']
result_dir = '/Users/long/Documents/data/gesture/'# temp data dir
save_dir=result_dir+'training_result/asFilter/'
Fs=200

############ test on different layers ########################
layer_index=1 #1 to xxx, loop through all model layers
for fi in range(outs[layer_index][0].shape[1]):
    outF=[outi[:,fi,:] for outi in outs[layer_index]]
    yf = [fft2(outFi,axes=1) for outFi in outF] # (160, 123)
    xf = fftfreq(N, 1/Fs)[:N//2] # 61
    yf2=[2.0 / N * np.abs(yfi[:,0:N // 2]) for yfi in yf] # (160, 61)
    yf_mean=[np.mean(tmp,axis=0) for tmp in yf2]
    yf_std = [np.std(tmp, axis=0) for tmp in yf2]
    for cl in range(5):
        ax.plot(xf,yf_mean[cl],color=colors[cl])
        ax.fill_between(xf,yf_mean[cl]-yf_std[cl],yf_mean[cl]+yf_std[cl],color=colors[cl],alpha=0.2)
    filename=save_dir+str(fi)+'.pdf'
    fig.savefig(filename)
    ax.clear()
