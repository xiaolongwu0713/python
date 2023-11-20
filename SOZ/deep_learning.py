import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from SOZ.util import get_channelwise_data_label, test_net
from common_dl import *
from common_plot import *


from SOZ.config import *
from dsp.common_dsp import identify_noisy_channel
from gesture.models.deepmodel import deepnet

if cuda:
    torch.backends.cudnn.benchmark = True

sf=1000
sid=2 # HJY
model_path=result_dir+'deepLearning/'
participant=participants[sid]
soz_channels=soz_channel_all[participant] # ['B1-4', 'C1-5']

## 1: non SOZ (label1=0);  2: SOZ (label2=1)
start=0
duration=1*60*60  # seconds
stop=start+duration # get partial data, or memory error will occure
wind=10*sf
data1, label1, data2, label2 = get_channelwise_data_label(sid,wind,start,stop) # (26280, 6000)
all_data=np.concatenate((data1[:,np.newaxis,:],data2[:,np.newaxis,:]),axis=0) # (52560,1, 6000)
all_labels=label1+label2 # 52560

X_train,X_val_test,y_train,y_val_test=train_test_split(all_data,all_labels,test_size=0.2, shuffle=True)
X_val,X_test,y_val,y_test=train_test_split(X_val_test,y_val_test,test_size=0.5, shuffle=True)
train_set=myDataset(X_train,np.asarray(y_train))
val_set=myDataset(X_val,np.asarray(y_val))
test_set=myDataset(X_test,np.asarray(y_test))
batch_size = 32
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)
train_size=len(train_loader.dataset)
val_size=len(val_loader.dataset)
test_size=len(test_loader.dataset)
n_chans=1
class_number=1 # one scalar for binary classification
wind=all_data.shape[2]
model_name='deepnet' # 50%
net = test_net(n_chans,class_number,wind).to(device) # 50%


def correct_count(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    #acc = correct_results_sum / y_test.shape[0]
    #acc = torch.round(acc * 100)
    return correct_results_sum

lr = 0.01
weight_decay = 1e-10
batch_size = 32
epoch_num = 500
patients=50
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr) # torch.optim.Adadelta(net.parameters(), lr=lr)

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
        optimizer.zero_grad()
        trainx = trainx.type(Tensor).to(device)
        y_pred = net(trainx) # torch.Size([32, 1, 3000])
        loss = criterion(y_pred.squeeze(), trainy.squeeze().type(Tensor).to(device))

        #print("Origin loss: "+ str(loss.item())+", regularization: "+ str(reg_variable)+".")
        loss=loss+reg_variable
        #print("New loss: " + str(loss.item()) + ".")
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() #* trainx.shape[0]
        running_corrects += correct_count(y_pred.cpu().squeeze(),trainy.squeeze())
    #print("train_size: " + str(train_size))
    #lr_scheduler.step() # test it
    train_loss = running_loss / train_size
    train_acc = (running_corrects / train_size).item()
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
                val_x = val_x.type(Tensor).to(device)
                outputs = net(val_x)
                running_corrects += correct_count(outputs.cpu().squeeze(), val_y.squeeze())
        val_acc = (running_corrects / val_size).item()
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


