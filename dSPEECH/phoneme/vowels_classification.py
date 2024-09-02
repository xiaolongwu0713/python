'''
This script is to discriminate between three vowels: iy/ae/uw;
Accuracy: 33%
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dSPEECH.config import *
from dSPEECH.phoneme.util import train_test_split, wind_list_of_2D
from gesture.models.d2l_resnet import d2lresnet
from gesture.utils import windowed_data

#device=torch.device('cpu')
modality='SEEG'
sid=2 # 1/2
sf=1024
result_dir = data_dir + 'processed/'+modality+str(sid)+'/CV/'
filename=result_dir+'vowel_epoch.npy'
tmp=np.load(filename,allow_pickle='TRUE').item()

epoch_iy=tmp['epoch_iy']
epoch_ae=tmp['epoch_ae']
epoch_uw=tmp['epoch_uw']

print("Standard scaler.")
scaler = StandardScaler()
epoch_iy=[scaler.fit_transform((tmp)) for tmp in epoch_iy]
epoch_ae=[scaler.fit_transform((tmp)) for tmp in epoch_ae]
epoch_uw=[scaler.fit_transform((tmp)) for tmp in epoch_uw]

wind=200
stride=10

train_iy,val_iy,test_iy=train_test_split(epoch_iy)
train_ae,val_ae,test_ae=train_test_split(epoch_ae)
train_uw,val_uw,test_uw=train_test_split(epoch_uw)

train_iy,val_iy,test_iy=wind_list_of_2D(train_iy),wind_list_of_2D(val_iy),wind_list_of_2D(test_iy)
train_ae,val_ae,test_ae=wind_list_of_2D(train_ae),wind_list_of_2D(val_ae),wind_list_of_2D(test_ae)
train_uw,val_uw,test_uw=wind_list_of_2D(train_uw),wind_list_of_2D(val_uw),wind_list_of_2D(test_uw)


X_train=np.asarray(train_iy+train_ae+train_uw)
X_val=np.asarray(val_iy+val_ae+val_uw)
X_test=np.asarray(test_iy+test_ae+test_uw)
y_train=[0,]*len(train_iy)+[1,]*len(train_ae)+[2,]*len(train_uw)
y_val=[0,]*len(val_iy)+[1,]*len(val_ae)+[2,]*len(val_uw)
y_test=[0,]*len(test_iy)+[1,]*len(test_ae)+[2,]*len(test_uw)

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

lr = 0.005
weight_decay = 1e-10
epoch_num = 100
patient=8

model_name='resnet'
class_number=3
net=d2lresnet(class_num=class_number,end_with_logsoftmax=False) # 92%
net=net.to(device)
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.BCEWithLogitsLoss()
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
        trainy = trainy.to(device)
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        preds = y_pred.argmax(dim=1, keepdim=True) # Returns the indices of the maximum value of all elements in the input tensor.
        #_, preds = torch.max(y_pred, 1)

        # if cuda:
        #     loss = criterion(y_pred, trainy.squeeze().cuda().long())
        # else:
        #     loss = criterion(y_pred, trainy.squeeze())
        loss = criterion(y_pred, trainy.squeeze())
        #print("Origin loss: "+ str(loss.item())+", regularization: "+ str(reg_variable)+".")
        loss=loss+reg_variable
        #print("New loss: " + str(loss.item()) + ".")
        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]
        running_corrects += torch.sum(preds.cpu().squeeze() == trainy.cpu().squeeze())
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
                val_y = val_y.to(device)
                outputs = net(val_x)
                #_, preds = torch.max(outputs, 1)
                preds = outputs.argmax(dim=1, keepdim=True)
                running_corrects += torch.sum(preds.cpu().squeeze() == val_y.cpu().squeeze())

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
        savepath = result_dir + 'checkpoint_'+model_name+'_' + str(best_epoch) + '_vowel.pth'
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
        test_y = test_y.to(device)
        outputs = net(test_x)
        #_, preds = torch.max(outputs, 1)
        preds = outputs.argmax(dim=1, keepdim=True)
        running_corrects += torch.sum(preds.cpu().squeeze() == test_y.cpu().squeeze())

test_acc = (running_corrects.double() / test_size).item()
print("Test accuracy: {:.2f}.".format(test_acc))

filename=result_dir +  model_name + '_'+ str(wind)+'_'+str(stride) + '_test_acc_vowel.txt'
with open(filename,'w') as f:
    f.write("Test accuracy: {:.2f}.".format(test_acc))
    f.write('\n')

train_result={}
train_result['train_losses']=train_losses
train_result['train_accs']=train_accs
train_result['val_accs']=val_accs
train_result['test_acc']=test_acc

filename=result_dir +  model_name + '_'+ str(wind)+'_'+str(stride) + '_vowel.npy'
np.save(filename,train_result)

#load
#train_result = np.load(filename,allow_pickle='TRUE').item()
#print(read_dictionary['train_losses'])




