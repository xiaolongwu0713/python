# The testing accuracy doesn't change at all. Don't understand.

#%cd /content/drive/MyDrive/
# raw_data is imported from global config

#%%capture
#! pip install mne==0.19.2;
#! pip install torch==1.7.0;
#! pip install hdf5storage;
#! pip install skorch==0.10.0;
#! pip install Braindecode==0.5.1;



import hdf5storage
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torch.optim import lr_scheduler
from torch import nn
from sklearn.metrics import accuracy_score
from gesture.models.resnet18 import _my_resnet18, my_resnet18
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier

from gesture.config import *
from common_dl import myDataset, set_random_seeds
from myskorch import on_epoch_begin_callback, on_batch_end_callback
from common_dl import count_parameters


from gesture.models.resnet18 import my_resnet18

sid = 10  # 4
chn_num=10
data_dir = data_dir + 'preprocessing/P' + str(sid) + '/tfInput/'
filename=data_dir+'dataset_'+str(chn_num)+'chn.hdf5'
f1 = h5py.File(filename, "r")
list(f1.keys())
X_train = f1['X_train'][:] # (650, 10, 148, 250)
X_test = f1['X_test'][:] #(50, 10, 148, 250)
y_train = f1['y_train'][:] # (650,)
y_test = f1['y_test'][:]
f1.close()

mean=X_train.mean()
std=X_train.std()
X_train=(X_train-mean)/std
X_test=(X_test-mean)/std


batch_size=batch_size
train_ds = myDataset(X_train, y_train)
val_ds = myDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, pin_memory=False)
#test it
#(x,y)=iter(train_loader).next() # x: torch.Size([1, 10, 148, 250]); y: torch.Size([1])


X_test.shape

len(val_loader)*4

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
seed = 20200220
set_random_seeds(seed=seed) # same as braindecode random seeding

net=my_resnet18(chn_num,5, pretrained=False,logsoftmax=False).float()

if cuda:
    net.cuda()
#x= torch.randn(1, 10, 148, 250)
#net(x).shape

lr = 0.0001
weight_decay = 1e-10
batch_size = 4
epoch_num = 500

#criterion=torch.nn.NLLLoss()
#optimizer = torch.optim.Adam(net.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#count_parameters(net)

epoch_num = 500

for epoch in range(epoch_num):
    print("------ epoch " + str(epoch) + " -----")
    net.train()

    loss_epoch = 0
    # trial=0
    target=[]
    predict=[]
    
    running_loss = 0.0
    running_corrects = 0
    for _, (trainx, trainy) in enumerate(train_loader):
        optimizer.zero_grad()
        if (cuda):
            trainx = trainx.float().cuda()
            #trainy = trainy.float().cuda()
        else:
            #pass
            trainx = trainx.float()
            #target = trainy.float()
        y_pred = net(trainx)
        _, preds = torch.max(y_pred, 1)
        #tmp=[a.tolist().index(max(a.tolist())) for a in y_pred.detach().cpu().numpy()]
        #predict=predict+tmp
        target=target+trainy.numpy().tolist()
        # target = torch.from_numpy(target)

        # regularization
        if cuda:
            loss = criterion(y_pred, trainy.type(torch.LongTensor).cuda())
        else:
            loss = criterion(y_pred, trainy.type(torch.LongTensor))
        
        #w0=net.model.layer1[0].conv1.weight.clone()
        loss.backward() # calculate the gradient and store in .grad attribute.
        optimizer.step()
        #w1=net.model.layer1[0].conv1.weight.clone()
        #grad=net.model.layer1[0].conv1.weight.grad
        #w2=w0+grad
        #print(torch.equal(w0,w1)) #return false
        #print(torch.equal(w1,w2)) #return false
        
        #b=list(net.named_parameters())
        #a=[torch.count_nonzero(i[1].grad).gt(0).item() for i in b] # true means its grad != 0
        #print(any(a)) # seems like there are grad except for the first two layers, why?
        
        running_loss += loss.item() * trainx.size(0)
        running_corrects += torch.sum(preds.cpu() == trainy.data)

    lr_scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects.double() / len(train_loader)
    print("Epoch " + str(epoch) + ": loss: " + str(epoch_loss) + ","+"Accuracy: " + str(epoch_acc.item()) + ".")
    
    running_loss = 0.0
    running_corrects = 0
    if epoch % 1 == 0:
        net.eval()
        #print("Validating...")
        with torch.no_grad():
            for _, (val_x, val_y) in enumerate(val_loader):
                if (cuda):
                    val_x = val_x.float().cuda()
                    #val_y = val_y.float().cuda()
                else:
                    val_x = val_x.float()
                    #val_y = val_y.float()
                outputs = net(val_x)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds.cpu() == val_y.data)
            
        epoch_acc = running_corrects.double() / (len(val_loader)*batch_size)
        print("Evaluation accuracy: " + str(running_corrects.item()) + '/('+ str(len(val_loader)) + '*'+str(batch_size)+')='+ str(epoch_acc.item()) + ".")



trainx, trainy =iter(train_loader).next()
y_pred = net(trainx.float().cuda())
loss = criterion(y_pred, trainy.type(torch.LongTensor).cuda())
loss.backward() # calculate the gradient and store in .grad attribute.
optimizer.step()

b=list(net.named_parameters())

[torch.count_nonzero(i[1].grad).gt(0).item() for i in b] # false means grad is zero

[bi[0] for bi in b if not bi[1].requires_grad]
