import sys
import socket

from tqdm import tqdm

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])

from gesture.config import *

import os, re
import numpy as np
from scipy.io import savemat
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from braindecode.datasets import (create_from_mne_raw, create_from_mne_epochs)
import torch
import timm
from common_dl import set_random_seeds
from common_dl import myDataset
from comm_utils import slide_epochs
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from braindecode.models import ShallowFBCSPNet,EEGNetv4,Deep4Net

from gesture.models.deepmodel import deepnet,deepnet_seq,deepnet_rnn, deepnet_da,deepnet_changeDepth,deepnet_expandPlan
from gesture.models.d2l_resnet import d2lresnet
from gesture.models.deepmodel import TSception2
from gesture.config import *

seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

model_name = 'resnet' #'deepnet'
fs=1000
wind = 500
stride = 200
class_number=2
#Session_num,UseChn,EmgChn,TrigChn = get_channel_setting(sid)
#fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]
result_path = 'H:/Long/data/sEMG_Zhangbin/training/'
data_dir = 'H:/Long/data/sEMG_Zhangbin/'

# shape: (trial_number, window_size, channels)
[ET_train,PD_train,others_train,NC_train] = np.load(data_dir+'dataset_train.npy', allow_pickle=True)
[ET_val,PD_val,others_val,NC_val] = np.load(data_dir+'dataset_val.npy', allow_pickle=True)
[ET_test,PD_test,others_test,NC_test] = np.load(data_dir+'dataset_test.npy', allow_pickle=True)

ET_train_y=[0]*len(ET_train)
ET_val_y=[0]*len(ET_val)
ET_test_y=[0]*len(ET_test)

PD_train_y=[0]*len(PD_train)
PD_val_y=[0]*len(PD_val)
PD_test_y=[0]*len(PD_test)

others_train_y=[0]*len(others_train)
others_val_y=[0]*len(others_val)
others_test_y=[0]*len(others_test)


NC_train_y=[0]*len(NC_train)
NC_val_y=[0]*len(NC_val)
NC_test_y=[0]*len(NC_test)

'''
ET_data=np.load(data_dir+'ET_data_3D.npy').transpose(0,2,1)
PD_data=np.load(data_dir+'PD_data_3D.npy').transpose(0,2,1)
others_data=np.load(data_dir+'others_data_3D.npy').transpose(0,2,1)
NC_data=np.load(data_dir+'NC_data_3D.npy').transpose(0,2,1)


ET_train,ET_val_test,ET_train_y,ET_val_test_y=train_test_split(ET_data,[0]*ET_data.shape[0],test_size=0.4,random_state=42)
ET_val,ET_test,ET_val_y,ET_test_y=train_test_split(ET_val_test,ET_val_test_y,test_size=0.5,random_state=42)
PD_train,PD_val_test,PD_train_y,PD_val_test_y=train_test_split(PD_data,[1]*PD_data.shape[0],test_size=0.4,random_state=42)
PD_val,PD_test,PD_val_y,PD_test_y=train_test_split(PD_val_test,PD_val_test_y,test_size=0.5,random_state=42)
others_train,others_val_test,others_train_y,others_val_test_y=train_test_split(others_data,[2]*others_data.shape[0],test_size=0.4,random_state=42)
others_val,others_test,others_val_y,others_test_y=train_test_split(others_val_test,others_val_test_y,test_size=0.5,random_state=42)
NC_train,NC_val_test,NC_train_y,NC_val_test_y=train_test_split(NC_data,[3]*NC_data.shape[0],test_size=0.4,random_state=42)
NC_val,NC_test,NC_val_y,NC_test_y=train_test_split(NC_val_test,NC_val_test_y,test_size=0.5,random_state=42)

'''


X_train=np.concatenate((ET_train,PD_train,others_train,NC_train))
y_train=ET_train_y+PD_train_y+others_train_y+NC_train_y
X_val=np.concatenate((ET_val,PD_val,others_val,NC_val))
y_val=ET_val_y+PD_val_y+others_val_y+NC_val_y
X_test=np.concatenate((ET_test,PD_test,others_test,NC_test))
y_test=ET_test_y+PD_test_y+others_test_y+NC_test_y

weights=[X_train.shape[0] / datai.shape[0] for datai in [ET_train, PD_train, others_train, NC_train]]
#weight = 1. / class_sample_count
samples_weight = np.array([weights[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_set=myDataset(X_train,np.asarray(y_train)) #len(train_set):217809
val_set=myDataset(X_val,np.asarray(y_val)) #len(val_set):72603
test_set=myDataset(X_test,np.asarray(y_test)) #len(test_set):72606

batch_size = 320
train_loader = DataLoader(dataset=train_set, batch_size=batch_size,pin_memory=False,sampler=sampler)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)
train_size=len(train_loader.dataset)  #1520
val_size=len(val_loader.dataset) # 190
test_size=len(test_loader.dataset) #190
## test the sampler
#_,yy=next(iter(train_loader))
#dict(zip(list(yy.numpy()),[list(yy.numpy()).count(i) for i in list(yy.numpy())]))
# Extract number of chans and time steps from dataset
one_window=next(iter(train_loader))[0]
n_chans = one_window.shape[1]
input_window_samples=one_window.shape[2]

#model_name='deepnet'
if model_name=='eegnet':
    #print('Here')
    net = EEGNetv4(n_chans, class_number, input_window_samples=input_window_samples, final_conv_length='auto', drop_prob=0.5)
elif model_name=='shallowFBCSPnet':
    net = ShallowFBCSPNet(n_chans,class_number,input_window_samples=input_window_samples,final_conv_length='auto',) # 51%
elif model_name=='deepnet':
    net = deepnet(n_chans,class_number,wind) # 81%
elif model_name=='deepnet_changeDepth':
    depth=4
    net = deepnet_changeDepth(n_chans,class_number,wind,depth) # 81%
    model_name='deepnet_changeDepth_'+str(depth)
elif model_name == 'deepnet2':
    net = deepnet_seq(n_chans, class_number, wind)
elif model_name == 'deepnet_rnn':
    net = deepnet_rnn(n_chans, class_number, wind)  # 65%
elif model_name=='resnet':
    net=d2lresnet(class_number,as_DA_discriminator=False) # 92%
elif model_name=='tsception':
    net = TSception2(1000, n_chans, 3, 3, 0.5)
elif model_name=='deepnet_da':
    net = deepnet_da(n_chans, class_number, wind)

#net = deepnet_resnet(n_chans,n_classes,input_window_samples=input_window_samples,expand=True) # 50%
#net=TSception(208)
#net=TSception(1000,n_chans,3,3,0.5)

lr = 0.01
weight_decay = 1e-10
batch_size = 32
patients=9999

img_size=[n_chans,wind]
#net = timm.create_model('visformer_tiny',num_classes=n_classes,in_chans=1,img_size=img_size)
net=net.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
#criterion = nn.NLLLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)
#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# Decay LR by a factor of 0.1 every 7 epochs
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

train_losses=[]
train_accs=[]
val_accs=[]
reg_type= torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
for epoch in range(5):
    print("------ epoch " + str(epoch)+" -----")
    net.train()
    loss_epoch = 0
    reg_variable=reg_type([0])
    running_loss = 0.0
    running_corrects = 0
    for batch, (trainx, trainy) in enumerate(tqdm(train_loader)):
        if isinstance(net, timm.models.visformer.Visformer):
            trainx=torch.unsqueeze(trainx,dim=1)
        optimizer.zero_grad()
        if (cuda):
            trainx = trainx.float().cuda()
        else:
            trainx = trainx.float()
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        preds = y_pred.argmax(dim=1, keepdim=True) # Returns the indices of the maximum value of all elements in the input tensor.
        #_, preds = torch.max(y_pred, 1)

        if cuda:
            loss = criterion(y_pred, trainy.squeeze().cuda().long())
        else:
            loss = criterion(y_pred, trainy.squeeze().long())

        if model_name != 'resnet':
            for i, layer in enumerate(net.layers):
                reg_variable = reg_variable+torch.sum(torch.pow(layer.weight.detach(), 2))
            reg_variable = weight_decay * reg_variable
        #print("Origin loss: "+ str(loss.item())+", regularization: "+ str(reg_variable)+".")
        loss=loss+reg_variable
        #print("New loss: " + str(loss.item()) + ".")
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]
        running_corrects += torch.sum(preds.cpu().squeeze() == trainy.squeeze())
        #break
    #print("train_size: " + str(train_size))
    #lr_scheduler.step() # test it
    train_loss = running_loss / train_size
    train_acc = (running_corrects.double() / train_size).item()
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    #print("Training loss: {:.2f}; Accuracy: {:.2f}.".format(train_loss,train_acc))
    #print("Training " + str(epoch) + ": loss: " + str(epoch_loss) + "," + "Accuracy: " + str(epoch_acc.item()) + ".")

    running_loss = 0.0
    running_corrects = 0
    if epoch % 1 == 0:
        net.eval()
        print("Validating...")
        with torch.no_grad():
            for _, (val_x, val_y) in enumerate(tqdm(val_loader)):
                if isinstance(net, timm.models.visformer.Visformer):
                    val_x = torch.unsqueeze(val_x, dim=1)
                if (cuda):
                    val_x = val_x.float().cuda()
                    # val_y = val_y.float().cuda()
                else:
                    val_x = val_x.float()
                    # val_y = val_y.float()
                outputs = net(val_x)
                #_, preds = torch.max(outputs, 1)
                preds = outputs.argmax(dim=1, keepdim=True)

                running_corrects += torch.sum(preds.cpu().squeeze() == val_y.squeeze())
                #break
        val_acc = (running_corrects.double() / val_size).item()
        val_accs.append(val_acc)
        print("Training loss:{:.2f},Accuracy:{:.2f}; Evaluation accuracy:{:.2f}.".format(train_loss, train_acc,val_acc))
    if epoch==0:
        best_epoch=0
        best_acc=val_acc
        patient=patients
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            # 'loss': epoch_loss
        }
    else:
        if val_acc>best_acc:
            best_epoch=epoch
            best_acc=val_acc
            patient=patients
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                #'loss': epoch_loss
            }

        else:
            patient=patient-1
    print("patients left: {:d}".format(patient))

savepath = result_path + 'checkpoint_'+model_name+'_' + str(best_epoch) + '.pth'
torch.save(state, savepath)

checkpoint = torch.load(savepath)
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
# print("Validating...")
with torch.no_grad():
    running_corrects = 0
    for _, (test_x, test_y) in enumerate(tqdm(test_loader)):
        if isinstance(net, timm.models.visformer.Visformer):
            test_x = torch.unsqueeze(test_x, dim=1)
        if (cuda):
            test_x = test_x.float().cuda()
            # val_y = val_y.float().cuda()
        else:
            test_x = test_x.float()
            # val_y = val_y.float()
        outputs = net(test_x)
        #_, preds = torch.max(outputs, 1)
        preds = outputs.argmax(dim=1, keepdim=True)

        running_corrects += torch.sum(preds.cpu().squeeze() == test_y.squeeze())
test_acc = (running_corrects.double() / test_size).item()
print("Test accuracy: {:.2f}.".format(test_acc))

train_result={}
train_result['train_losses']=train_losses
train_result['train_accs']=train_accs
train_result['val_accs']=val_accs
train_result['test_acc']=test_acc

#filename=result_path +  model_name + '_'+ str(wind)+'_'+str(stride) + '.npy'
#np.save(filename,train_result)

#load
#train_result = np.load(filename+'.npy',allow_pickle='TRUE').item()
#print(read_dictionary['train_losses'])
