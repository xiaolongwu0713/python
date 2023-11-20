import sys
import socket
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
from sklearn.model_selection import StratifiedKFold
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

from gesture.feature_selection.utils import get_good_sids, get_final_good_sids, get_selected_channel_gumbel, \
    get_selected_channel_stg
from gesture.utils import *
from gesture.models.deepmodel import deepnet,deepnet_seq,deepnet_rnn, deepnet_da,deepnet_changeDepth,deepnet_expandPlan
from gesture.models.d2l_resnet import d2lresnet
from gesture.models.deepmodel import TSception2
from gesture.config import *
from gesture.preprocess.chn_settings import get_channel_setting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

if 'PYTHONPATH' in os.environ:
    running_from_IDE=True
else:
    running_from_CMD = True

#if len(sys.argv)>3:
if running_from_CMD:
    sid = int(float(sys.argv[1]))
    model_name = sys.argv[2]
    fs = int(float(sys.argv[3]))
    wind = int(float(sys.argv[4]))
    stride = int(float(sys.argv[5]))
    try:
        retrain_use_selected_channel = int(float(sys.argv[6]))
    except IndexError:
        pass
else: # debug in IDE
    sid=10
    fs=1000
    wind = 500
    stride = 100
    model_name='resnet'
    retrain_use_selected_channel=False
class_number=5
#Session_num,UseChn,EmgChn,TrigChn = get_channel_setting(sid)
#fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]

#retrain_use_selected_channel=True

if retrain_use_selected_channel:
    channel_selection_method = 'stg'  # 'gumble'/'stg'
    result_path = result_dir + 'deepLearning/selected_channel_' + channel_selection_method + '/' + str(sid) + '/'
    model_path = result_path
    # selected channels
    if channel_selection_method == 'gumble':
        channel_num_selected = 10
        selected_channels, acc = get_selected_channel_gumbel(sid, channel_num_selected, ploty=False)
    elif channel_selection_method == 'stg':
        selected_channels, acc = get_selected_channel_stg(sid, ploty=False)
        selected_channels = selected_channels.tolist()
else:
    result_path = result_dir + 'deepLearning/' + str(sid) + '/'
    model_path=result_path

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

test_epochs, val_epochs, train_epochs=read_data(sid,fs,retrain_use_selected_channel, selected_channels)

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
    net = deepnet_changeDepth(n_chans,class_number,wind,depth) # 81%
    model_name='deepnet_changeDepth_'+str(depth)
elif model_name == 'deepnet2':
    net = deepnet_seq(n_chans, class_number, wind, )
elif model_name == 'deepnet_rnn':
    net = deepnet_rnn(n_chans, class_number, wind, )  # 65%
elif model_name=='resnet':
    net=d2lresnet() # 92%
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
epoch_num = 500
patients=20

img_size=[n_chans,wind]
#net = timm.create_model('visformer_tiny',num_classes=n_classes,in_chans=1,img_size=img_size)
if cuda:
    net.cuda()

criterion = torch.nn.CrossEntropyLoss()
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
for epoch in range(epoch_num):
    print("------ epoch" + str(epoch) + ": sid"+str(sid)+"@"+model_name+"-----")
    net.train()
    loss_epoch = 0
    reg_variable=reg_type([0])
    running_loss = 0.0
    running_corrects = 0
    for batch, (trainx, trainy) in enumerate(train_loader):
        if isinstance(net, timm.models.visformer.Visformer):
            trainx=torch.unsqueeze(trainx,dim=1)
        optimizer.zero_grad()
        if (cuda):
            trainx = trainx.float().cuda()
        else:
            trainx = trainx.float()
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        preds = y_pred.argmax(dim=1, keepdim=True)
        #_, preds = torch.max(y_pred, 1)

        if cuda:
            loss = criterion(y_pred, trainy.squeeze().cuda().long())
        else:
            loss = criterion(y_pred, trainy.squeeze())

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
        # print("Validating...")
        with torch.no_grad():
            for _, (val_x, val_y) in enumerate(val_loader):
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
    if patient==0:
        savepath = model_path + 'checkpoint_'+model_name+'_' + str(best_epoch) + '.pth'
        torch.save(state, savepath)

        break

checkpoint = torch.load(savepath)
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
# print("Validating...")
with torch.no_grad():
    running_corrects = 0
    for _, (test_x, test_y) in enumerate(test_loader):
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

#filename=result_path + 'training_result_'+model_name

filename=result_path + 'training_result_'+model_name+str(sid)+'_'+str(wind)
np.save(filename,train_result)

#load
#train_result = np.load(filename+'.npy',allow_pickle='TRUE').item()
#print(read_dictionary['train_losses'])

