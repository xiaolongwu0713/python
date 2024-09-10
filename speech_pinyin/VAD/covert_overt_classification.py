'''
This script is to discriminate between overt vs covert speech.
TODO: accuracy only changes a small value during training. Try to normalize the data.
'''
import glob

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from speech_pinyin.config import *
from dSPEECH.phoneme.util import train_test_split, wind_list_of_2D
from gesture.models.d2l_resnet import d2lresnet
from gesture.utils import windowed_data

#device=torch.device('cpu')
modality='SEEG'
sid=1 # 1/2
session=3
sf=1000

folder=data_dir+str(sid)+'-*'
folder=os.path.normpath(glob.glob(folder)[0])
folder=folder.replace("\\", "/")
result_dir=folder+'/result/'
filename=folder+'/processed/session'+str(session)+'_covert_overt_EEG.npy'
tmp=np.load(filename,allow_pickle='TRUE').item()
ons=tmp['overt']
offs=tmp['covert']

wind=100
stride=20
train_on1,val_on1,test_on1=train_test_split(ons)
train_off1,val_off1,test_off1=train_test_split(offs)
train_on,val_on,test_on=wind_list_of_2D(train_on1),wind_list_of_2D(val_on1),wind_list_of_2D(test_on1)
train_off,val_off,test_off=wind_list_of_2D(train_off1),wind_list_of_2D(val_off1),wind_list_of_2D(test_off1)
X_train=np.asarray(train_on+train_off)
X_val=np.asarray(val_on+val_off)
X_test=np.asarray(test_on+test_off)
y_train=[0,]*len(train_on)+[1,]*len(train_off)
y_val=[0,]*len(val_on)+[1,]*len(val_off)
y_test=[0,]*len(test_on)+[1,]*len(test_off)

train_set=myDataset(X_train,y_train)
val_set=myDataset(X_val,y_val)
test_set=myDataset(X_test,y_test)

batch_size = 40 # larger batch_size slows the training
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)

train_size=len(train_loader.dataset)  #1520
val_size=len(val_loader.dataset) # 190
test_size=len(test_loader.dataset) #190

# Extract number of chans and time steps from dataset
one_window=next(iter(train_loader))[0]
n_chans = one_window.shape[1]
input_window_samples=one_window.shape[2]

lr = 0.5
weight_decay = 1e-10
epoch_num = 500
patient=10

model_name='resnet'
class_number=2
net=d2lresnet(class_num=class_number,end_with_logsoftmax=False) # 92%
net=net.to(device)
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)

def correct_count(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    #acc = correct_results_sum / y_test.shape[0]
    #acc = torch.round(acc * 100)
    return correct_results_sum

train_losses=[]
train_accs=[]
val_accs=[]
reg_type= torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
for epoch in range(epoch_num):
    print("------ epoch" + str(epoch) + ': sid' +str(sid)+' using '+model_name+'-----')
    net.train()
    loss_epoch = 0
    reg_variable=reg_type([0])
    running_loss = 0.0
    running_corrects = 0
    for batch, (trainx, trainy) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        trainx = trainx.float().to(device)
        trainy = trainy.float().to(device)
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        #preds = y_pred.argmax(dim=1, keepdim=True) # Returns the indices of the maximum value of all elements in the input tensor.
        #_, preds = torch.max(y_pred, 1)

        # if cuda:
        #     loss = criterion(y_pred, trainy.squeeze().cuda().long())
        # else:
        #     loss = criterion(y_pred, trainy.squeeze())
        loss = criterion(y_pred, trainy.squeeze())
        #print("Origin loss: "+ str(loss.item())+", regularization: "+ str(reg_variable)+".")
        #loss=loss+reg_variable
        #print("New loss: " + str(loss.item()) + ".")
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]
        running_corrects += correct_count(y_pred.squeeze(),trainy.squeeze()) #torch.sum(preds.squeeze() == trainy.squeeze())
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
            for _, (val_x, val_y) in enumerate(tqdm(val_loader)):
                val_x = val_x.float().to(device)
                val_y = val_y.float().to(device)
                outputs = net(val_x)
                #_, preds = torch.max(outputs, 1)
                #preds = outputs.argmax(dim=1, keepdim=True)

                running_corrects += correct_count(outputs.squeeze(),val_y.squeeze())#torch.sum(preds.squeeze() == val_y.squeeze())

        val_acc = (running_corrects.double() / val_size).item()
        val_accs.append(val_acc)
        print("Training loss:{:.2f},Accuracy:{:.2f}; Evaluation accuracy:{:.2f}.".format(train_loss, train_acc,val_acc))
    if epoch==0:
        best_epoch=0
        best_acc=val_acc
        patient=patient
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
            patient=patient
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
        savepath = result_dir + 'checkpoint_'+model_name+'_' + str(best_epoch) + '.pth'
        torch.save(state, savepath)
        break

checkpoint = torch.load(savepath)
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
# print("Validating...")
with torch.no_grad():
    running_corrects = 0
    for _, (test_x, test_y) in enumerate(tqdm(test_loader)):
        test_x = test_x.float().to(device)
        test_y = test_y.float().to(device)
        outputs = net(test_x)
        #_, preds = torch.max(outputs, 1)
        #preds = outputs.argmax(dim=1, keepdim=True)

        running_corrects += correct_count(outputs.squeeze(),test_y.squeeze())# torch.sum(preds.squeeze() == test_y.squeeze())
test_acc = (running_corrects.double() / test_size).item()
print("Test accuracy: {:.2f}.".format(test_acc))

filename=result_dir +  model_name + '_'+ str(wind)+'_'+str(stride) + '_test_acc.txt'
with open(filename,'w') as f:
    f.write("Test accuracy: {:.2f}.".format(test_acc))
    f.write('\n')

train_result={}
train_result['train_losses']=train_losses
train_result['train_accs']=train_accs
train_result['val_accs']=val_accs
train_result['test_acc']=test_acc

filename=result_dir +  model_name + '_'+ str(wind)+'_'+str(stride) + '.npy'
np.save(filename,train_result)

#load
#train_result = np.load(filename,allow_pickle='TRUE').item()
#print(read_dictionary['train_losses'])




