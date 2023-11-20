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
from torch import nn
from gesture.models.resnet18 import my_resnet18
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier

from gesture.config import *
from common_dl import myDataset, set_random_seeds
from myskorch import on_epoch_begin_callback, on_batch_end_callback


from gesture.models.resnet18 import my_resnet18

sid = 10  # 4

data_dir = data_dir + 'preprocessing/P' + str(sid) + '/tfInput/'
filename=data_dir+'dataset.hdf5'
f1 = h5py.File(filename, "r")
list(f1.keys())
X_train = f1['X_train'][:] # (650, 10, 148, 250)
X_test = f1['X_test'][:] #(50, 10, 148, 250)
y_train = f1['y_train'][:] # (650,)
y_test = f1['y_test'][:]
f1.close()

train_ds = myDataset(X_train, y_train)
val_ds = myDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_ds, batch_size=1, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_ds, batch_size=1, pin_memory=False)
#test it
#(x,y)=iter(train_loader).next() # x: torch.Size([1, 10, 148, 250]); y: torch.Size([1])


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed) # same as braindecode random seeding

model=my_resnet18(10,5, pretrained=True)
if cuda:
    model.cuda()
#x= torch.randn(1, 10, 148, 250)
#model(x).shape

lr = 0.0001
weight_decay = 1e-10
batch_size = 1
n_epochs = 100

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
    criterion=torch.nn.NLLLoss,  #torch.nn.NLLLoss/CrossEntropyLoss
    optimizer=torch.optim.Adam, #optimizer=torch.optim.AdamW,
    train_split=predefined_split(val_ds),  # using valid_set for validation; None means no validate:both train and test on training dataset.
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=my_callbacks,
    device=device,
)

clf.fit(train_ds, y=None, epochs=n_epochs)


